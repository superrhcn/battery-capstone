"""Prepare/apply one-instance EC2 stop notifications. No EC2 role or stored key.

Default --plan is offline when --account-id is supplied. --apply uses the caller's
normal AWS CLI identity, e.g. an authenticated CloudShell session. Does not start,
stop, terminate, or modify the EC2 instance.
"""
from __future__ import annotations
import argparse
import json
import re
import subprocess


def aws(region, *parts):
    completed = subprocess.run(["aws", "--region", region, *parts,
        "--output", "json", "--no-cli-pager"], check=True,
        capture_output=True, text=True, encoding="utf-8")
    return json.loads(completed.stdout) if completed.stdout.strip() else {}


def build_plan(account_id, region, instance_id, email, topic_name=None):
    if not re.fullmatch(r"[0-9]{12}", account_id):
        raise ValueError("account-id must have 12 digits")
    if not re.fullmatch(r"i-[0-9a-f]{8,17}", instance_id):
        raise ValueError("invalid instance-id")
    if not re.fullmatch(r"[a-z]{2}(?:-[a-z]+)+-[0-9]+", region):
        raise ValueError("invalid AWS region")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise ValueError("invalid email")
    name = topic_name or "capstone-long-stop-" + instance_id[-8:]
    if not re.fullmatch(r"capstone-[A-Za-z0-9_-]{1,180}", name):
        raise ValueError("topic name must start capstone- and use safe characters")
    rule = name
    topic_arn = f"arn:aws:sns:{region}:{account_id}:{name}"
    rule_arn = f"arn:aws:events:{region}:{account_id}:rule/{rule}"
    pattern = {"source": ["aws.ec2"],
        "detail-type": ["EC2 Instance State-change Notification"],
        "account": [account_id], "region": [region],
        "detail": {"instance-id": [instance_id], "state": ["stopped", "terminated"]}}
    statement = {"Sid": "CapstoneEventBridgePublish", "Effect": "Allow",
        "Principal": {"Service": "events.amazonaws.com"},
        "Action": "sns:Publish", "Resource": topic_arn,
        "Condition": {"ArnEquals": {"aws:SourceArn": rule_arn}}}
    message = ("[배터리 캡스톤] 계산 서버 상태: <state>. "
        "대상 <instance>, 시각 <eventTime> (UTC). "
        "이 메일은 서버가 중지되었다는 알림이며 실험 성공을 뜻하지 않습니다. "
        "EBS의 cloud_status.json / outcome.json에서 SOH 80% 도달, "
        "최대 EFC, 시간 상한, 실패를 구분해 확인하십시오. "
        "EC2 중지 후 컴퓨팅 사용료는 멈추지만 EBS 저장공간 비용은 남습니다.")
    target = {"Id": "CapstoneStopEmail", "Arn": topic_arn,
        "InputTransformer": {"InputPathsMap": {"instance": "$.detail.instance-id",
            "state": "$.detail.state", "eventTime": "$.time"},
            "InputTemplate": json.dumps(message)}}
    return {"account_id": account_id, "region": region, "instance_id": instance_id,
        "email": email, "topic_name": name, "topic_arn": topic_arn,
        "rule_name": rule, "rule_arn": rule_arn, "event_pattern": pattern,
        "publish_statement": statement, "target": target,
        "note": "No EC2 instance profile/IAM role is added. Confirm SNS subscription email."}


def inspect_subscription(region, subscription):
    """A Subscribe-returned ARN alone is never proof of email confirmation."""
    arn = subscription.get("SubscriptionArn") or ""
    result = {"SubscriptionArn": arn, "confirmed": False,
              "pending_confirmation": "unknown"}
    if not arn.startswith("arn:"):
        if "pending" in arn.lower():
            result["pending_confirmation"] = "true"
        return result
    attrs = aws(region, "sns", "get-subscription-attributes", "--subscription-arn", arn).get("Attributes", {})
    pending = str(attrs.get("PendingConfirmation", "unknown")).lower()
    result.update(pending_confirmation=pending, confirmed=(pending == "false"))
    return result


def subscription_status(plan):
    subs = aws(plan["region"], "sns", "list-subscriptions-by-topic", "--topic-arn", plan["topic_arn"])
    matches = [inspect_subscription(plan["region"], sub)
        for sub in subs.get("Subscriptions", [])
        if sub.get("Protocol") == "email" and sub.get("Endpoint") == plan["email"]]
    return {"topic_arn": plan["topic_arn"], "subscription": matches,
            "confirmed": any(sub["confirmed"] for sub in matches),
            "delivery_tested": False}


def apply_plan(plan):
    region = plan["region"]
    # Read before resource creation: also verifies selected account and instance.
    ec2 = aws(region, "ec2", "describe-instances", "--instance-ids", plan["instance_id"])
    instances = [i for r in ec2.get("Reservations", []) for i in r.get("Instances", [])]
    if len(instances) != 1:
        raise RuntimeError("Expected exactly one selected EC2 instance")
    created = aws(region, "sns", "create-topic", "--name", plan["topic_name"],
        "--tags", "Key=Purpose,Value=battery-capstone-stop-notification")
    if created.get("TopicArn") != plan["topic_arn"]:
        raise RuntimeError("Topic ARN differs from reviewed plan")
    attrs = aws(region, "sns", "get-topic-attributes", "--topic-arn", plan["topic_arn"])["Attributes"]
    if attrs.get("KmsMasterKeyId"):
        raise RuntimeError("Existing topic uses KMS; review EventBridge KMS permissions before proceeding")
    # Preserve unrelated permissions. Never remove another statement.
    policy = json.loads(attrs.get("Policy", '{"Version":"2012-10-17","Statement":[]}'))
    policy["Statement"] = [s for s in policy.get("Statement", [])
                           if s.get("Sid") != "CapstoneEventBridgePublish"]
    policy["Statement"].append(plan["publish_statement"])
    aws(region, "sns", "set-topic-attributes", "--topic-arn", plan["topic_arn"],
        "--attribute-name", "Policy", "--attribute-value", json.dumps(policy))
    rule = aws(region, "events", "put-rule", "--name", plan["rule_name"],
        "--description", "Battery capstone selected EC2 stopped/terminated email; not success proof",
        "--event-pattern", json.dumps(plan["event_pattern"]), "--state", "ENABLED")
    if rule.get("RuleArn") != plan["rule_arn"]:
        raise RuntimeError("Rule ARN differs from reviewed plan")
    targets = aws(region, "events", "put-targets", "--rule", plan["rule_name"],
        "--targets", json.dumps([plan["target"]]))
    if targets.get("FailedEntryCount", 0):
        raise RuntimeError("EventBridge target creation failed: " + json.dumps(targets))
    subs = aws(region, "sns", "list-subscriptions-by-topic", "--topic-arn", plan["topic_arn"])
    matches = [s for s in subs.get("Subscriptions", [])
               if s.get("Protocol") == "email" and s.get("Endpoint") == plan["email"]]
    if not matches:
        sub = aws(region, "sns", "subscribe", "--topic-arn", plan["topic_arn"],
            "--protocol", "email", "--notification-endpoint", plan["email"],
            "--return-subscription-arn")
        matches = [{"SubscriptionArn": sub.get("SubscriptionArn")}]
    inspected = [inspect_subscription(region, sub) for sub in matches]
    return {"topic_arn": plan["topic_arn"], "rule_arn": plan["rule_arn"],
        "subscription": inspected, "instance_state": instances[0]["State"]["Name"],
        "confirmed": any(sub["confirmed"] for sub in inspected),
        "delivery_tested": False,
        "note": "Configuration only. User must confirm pending email subscription; test delivery separately."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--account-id")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--topic-name")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true")
    action.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    account = args.account_id
    if args.apply or args.status_only:
        actual = aws(args.region, "sts", "get-caller-identity")["Account"]
        if account and account != actual:
            parser.error("Selected account differs from authenticated caller")
        account = actual
    elif not account:
        parser.error("Offline plan requires --account-id; --apply resolves authenticated account")
    plan = build_plan(account, args.region, args.instance_id, args.email, args.topic_name)
    result = apply_plan(plan) if args.apply else subscription_status(plan) if args.status_only else plan
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
