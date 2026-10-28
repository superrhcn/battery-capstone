# 장기 실험의 독립 실행·자동 중지·메일 알림

작성 기준: 2026-10-28. 이번 장기 예비실험의 대상은 **셀 4번·CC 3.2 A 한 조건**이다. 기존 네 전류 100 EFC와 별도 출력 폴더를 사용한다. 이 문서는 실행 제어 코드의 설계·검증 기록이며, AWS 장기 계산이나 실제 메일 수신을 완료했다는 기록이 아니다.

## 1. 실행 조건과 파일

| 항목 | 설정·역할 |
| --- | --- |
| 물리 계산 | `simulation/run_lifetime.py`; 셀 4, 3.2 A, 격자 ×4 |
| 종료 | SOH 80% 관측, 최대 3,000 EFC, 누적 궤적 실행시간 72,000 s 중 먼저 발생한 사유 기록 |
| 기준시험 | 100 EFC 이후에도 10 EFC 간격 유지; 종료 시점은 관측 구간으로 기록하며 중간 시각을 임의 보간하지 않음 |
| 서버 외부 감독 | `scripts/run_lifetime_cloud.py` |
| 독립 서비스 실행 | `scripts/launch_lifetime_service.sh` |
| 컴퓨팅 안전 상한 | 별도 systemd 타이머 72,900 s = 20시간 15분. 20시간 계산 예산 뒤 종료 처리 여유 15분 |
| 결과 경로 | `RUN_ROOT/trajectory/`에 물리 계산 결과; `RUN_ROOT/cloud_status.json`, `runner.log`, `cloud_supervisor.log`에 실행 관리 기록 |
| 알림 | 특정 EC2 인스턴스의 `stopped` 또는 `terminated` 사건 → EventBridge → SNS 이메일 |
| 비밀번호·키 | 코드나 결과에 저장하지 않음. EC2에 SNS 발행 권한도 추가하지 않음 |

장기 실행기는 기존 100 EFC의 상태·Q₀·누적 방전량을 검증해 불러온다. 원본 100 EFC 폴더를 덮어쓰지 않는다. 불완전한 방전 중간 상태의 재개 처리와 코드·환경 식별 검사는 물리 실행기가 담당한다.

## 2. 채팅을 꺼도 계산이 이어지는 구조

SSM 또는 관리 셸에서는 `systemd-run`으로 서비스를 **시작만** 한다. 계산 프로세스는 EC2의 systemd가 관리하므로 브라우저·채팅·사용자 PC가 종료되어도 해당 세션에 종속되지 않는다. 이 서비스는 매 부팅 자동 실행으로 등록하지 않는다. 운영자가 재개를 명시해야 한다.

서버 감독기는 별도 타이머를 먼저 등록하고 자식 실행기를 시작한다. 결과가 완료·실패·상한 종료 중 무엇이든 상태 JSON을 저장하고 디스크를 동기화한 뒤 OS 종료를 요청한다. 감독기 자체가 비정상 종료되면 서비스의 `ExecStopPost`가 1분 뒤 종료를 요청한다. 별도 20시간 15분 타이머는 마지막 안전 장치이다. 메일 전송이나 네트워크가 실패해도 종료 조건에는 영향을 주지 않는다.

RAM·디스크 감시로 중지할 때는 부모 PID 종료만 보지 않고 계산 프로세스 그룹 전체가 종료될 때까지 최대 75초 기다린다. 남은 프로세스가 있으면 SIGKILL 후 다시 확인하고 강제 종료 여부를 기록한다. 최종 콘솔과 관리 JSON에는 상태·마지막 EFC·SOH·도달 구간·파일 경로만 요약하고 전체 이력은 EBS 원본 파일로 보존한다.

중지 직전의 마지막 주기가 완성되지 않았으면, 이미 완료한 주기의 원자적 체크포인트까지만 재개할 수 있다. 장애가 발생한 마지막 조각을 완료한 것처럼 사용하지 않는다.

## 3. 서버에서 실행할 명령

아래 경로는 실행 방법을 보여주는 예시다. 새 코드의 고정 커밋, 실제 원본 폴더명, EBS 백업 검증이 끝난 뒤 경로를 확정한다. 기존 Python 가상환경과 별도 소스 폴더를 사용한다.

```bash
/var/lib/battery-capstone/venv/bin/python \
  /var/lib/battery-capstone/lifetime-repo/scripts/run_lifetime_cloud.py \
  --plan \
  --repo-root /var/lib/battery-capstone/lifetime-repo \
  --python /var/lib/battery-capstone/venv/bin/python \
  --run-root /var/lib/battery-capstone/results/lifetime_cell4_3p2A \
  --import-100efc-dir /var/lib/battery-capstone/results/main_100efc/current_3.2A
```

`--plan`은 명령과 시간 상한만 출력한다. 계산, 타이머 등록, 서버 중지를 하지 않는다. 아래 실제 실행 명령에는 `--plan`을 넣지 않는다.

```bash
CAPSTONE_REPO_ROOT=/var/lib/battery-capstone/lifetime-repo \
CAPSTONE_PYTHON=/var/lib/battery-capstone/venv/bin/python \
bash /var/lib/battery-capstone/lifetime-repo/scripts/launch_lifetime_service.sh \
  --run-root /var/lib/battery-capstone/results/lifetime_cell4_3p2A \
  --import-100efc-dir /var/lib/battery-capstone/results/main_100efc/current_3.2A \
  --max-wall-s 72000
```

기존 장기 폴더를 이어갈 때는 같은 코드·환경에서 `--import-100efc-dir ...` 대신 `--resume`을 사용한다. 기존 서비스나 타이머가 실행 중이면 새 작업을 겹쳐 시작하지 않는다.

확인할 명령은 다음과 같다.

```bash
systemctl status capstone-lifetime.service --no-pager
systemctl status capstone-lifetime-hard-stop.timer --no-pager
cat /var/lib/battery-capstone/results/lifetime_cell4_3p2A/cloud_status.json
```

서버 중지 뒤에는 저장공간 사용료가 남는다. 데이터 회수와 검증이 끝나기 전에 EBS를 삭제하지 않는다.

## 4. 이메일 설정과 수신 확인

`configure_stop_email.py`는 설정 계획을 출력하거나, 인증된 CloudShell의 AWS CLI로 SNS 주제·특정 인스턴스의 EventBridge 규칙·이메일 구독을 만든다. EC2 시작·중지 명령은 실행하지 않는다. 수신 주소는 CLI로 전달하며 공개 저장소에 하드코딩하지 않는다.

```text
python scripts/configure_stop_email.py --account-id YOUR_12_DIGIT_ACCOUNT --instance-id YOUR_INSTANCE_ID --email YOUR_EMAIL
python scripts/configure_stop_email.py --apply --instance-id YOUR_INSTANCE_ID --email YOUR_EMAIL
python scripts/configure_stop_email.py --status-only --instance-id YOUR_INSTANCE_ID --email YOUR_EMAIL
```

`subscribe --return-subscription-arn`은 확인 전에도 ARN을 반환한다. ARN 존재만으로 확인 완료라고 판정하지 않는다. 설정 도구는 `get-subscription-attributes`의 `PendingConfirmation`이 `false`인지를 조회한다. 읽기 전용 확인에는 `--status-only`를 사용한다.

실제 구독에는 AWS에서 보내는 **Subscription Confirmation** 메일을 사용자가 확인해야 한다. 상태가 `PendingConfirmation`이면 완료 메일 수신 준비가 끝난 것이 아니다. 구독 확인 뒤 별도의 시험 메일 또는 실제 서버 상태 전환으로 전달을 확인한다.

이 경로에는 EC2용 SNS IAM 권한이 필요 없다. SNS의 리소스 정책은 `events.amazonaws.com`의 `sns:Publish`만 허용하고, `aws:SourceArn`을 해당 규칙 하나로 제한한다. 기존의 다른 정책 문장은 보존한다. 기존 SNS 주제가 KMS로 암호화되어 있으면 별도의 키 권한 검토가 필요하므로 설정 도구가 멈춘다.

**메일은 서버가 멈췄다는 사실을 알린다.** SOH 80% 도달·3,000 EFC 도달·시간 상한·실패 여부는 `outcome.json`과 `cloud_status.json`에서 확인해야 한다. 수동 중지나 외부 장애도 같은 서버 중지 메일을 발생시킬 수 있다. 메일이 오지 않았다는 이유만으로 서버가 계속 실행 중이라고 단정하지 않는다.

## 5. 실행시간과 비용을 읽는 기준

앞선 네 조건 병렬 계산에서 3.2 A의 100 EFC 실행시간은 약 40.49분이었다. 같은 속도가 유지된다고만 가정하면 0→3,000 EFC는 약 20.25시간, 저장한 100 EFC 이후 2,900 EFC는 약 19.57시간이다. 기준시험은 10 EFC 간격을 유지하며 3,000 EFC까지 최대 302개 시점(0·5·10·…·3,000)이 있다. 이 값은 선형 환산이며 장기 열화에 따른 솔버 비용과 단독 실행 속도 차이를 반영하지 않았다. 누적 20시간 상한에 먼저 걸려 3,000 EFC까지 못 갈 수 있다. 첫 장기 실행에서 실제 처리량을 다시 측정한다.

확인했던 컴퓨팅 단가 0.4032 USD/h를 유지한다고 가정하면 20시간은 약 8.064 USD, 서버 안전 상한 20시간 15분까지는 약 8.165 USD이다. EBS·공인 IPv4·전송·세금·환전·크레딧은 별도다. 이는 확정 청구액이 아니다. SOH 80%에 도달하지 않은 채 상한에 걸리면 검열된 관측값으로 남긴다.

## 6. 확인한 검사와 남은 확인

2026-10-28 로컬에서 Python 문법 검사와 오프라인 9개 검사를 통과하였다. 검사는 알림 대상 제한, 잘못된 입력 거부, 물리 실행기 CLI 연결, 정상 결과 후 종료 요청, 실행기 시작 실패 후 상태 저장·종료 요청을 확인하였다. 실제 AWS API·물리 계산·systemd·종료 호출은 모의 처리하였다. Windows 샌드박스의 파일 교체 제약 때문에 감독 흐름 검사에서는 JSON 저장도 모의 처리하였다. Linux의 원자적 파일 교체를 검증한 결과로 해석하지 않는다.

Linux에서 `bash -n scripts/launch_lifetime_service.sh`와 같은 오프라인 검사를 다시 실행해야 한다. 실제 서비스·타이머의 활성 상태, 원본 데이터 회수, 실제 메일 수신, EC2 중지 상태는 배포 시 별도 확인한다. 테스트가 통과했다는 사실만으로 배포 완료를 표시하지 않는다.

## 7. 공식 근거

- [EC2 상태 전환 사건](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/monitoring-instance-state-changes.html): `stopped`와 `terminated` 등의 사건 형식.
- [EC2 상태 전환 이메일 알림 예제](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-events-eventbridge-example.html): SNS 이메일 구독 확인과 EventBridge 연결.
- [EventBridge 리소스 기반 권한](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-use-resource-based.html): EC2 역할 없이 SNS 주제 정책으로 발행 허용.
- [AWS SAM이 생성하는 EventBridge→SNS 정책](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/reference-sam-connector.html): SNS 정책에 특정 규칙의 SourceArn 조건을 쓰는 공식 예제.
- [EC2 user data 실행 주기](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html): Linux user data는 기본 최초 실행 1회다. 기존 서버를 단순 재시작하거나 user data만 바꾼 것으로 새 작업이 실행된다고 보장하지 않는다.

- [SNS subscribe의 ARN 반환](https://docs.aws.amazon.com/cli/latest/reference/sns/subscribe.html): 확인 전에도 ARN을 반환할 수 있다.
- [SNS 구독 속성 조회](https://docs.aws.amazon.com/cli/latest/reference/sns/get-subscription-attributes.html): PendingConfirmation으로 확인 상태를 판정한다.
