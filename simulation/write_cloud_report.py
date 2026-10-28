"""Write a concise experiment report from recorded batch results only."""
from __future__ import annotations
from pathlib import Path


def write_report(out, batch, compact):
    lines = ['# 배터리 첫 100 EFC 실험 보고서', '',
        f"- 실행 상태: **{batch['status']}**",
        f"- 시작 시각(UTC): {batch['started_utc']}",
        f"- 실행 환경: {batch['platform']}; Python {batch['python']}",
        f"- CPU {batch['cpu_count']}개, OS 인식 RAM {batch['memory_mb']['MemTotal']/1024:.2f} GiB",
        f"- 배치 경과 시간: {batch.get('wall_s',0)/3600:.3f}시간",
        f"- 선택한 동시 실행 수: {batch.get('selected_workers','미선정')}", '',
        '## 실험 목적과 조건', '',
        '기준 셀 4번에서 고정 CC 전류 3.0·3.1·3.2·3.3 A를 비교한다. 각 조건의 첫 100 EFC 동안 SOH, 충전시간, 계산 비용을 측정한다. 100 EFC는 충방전 100회와 다르다.', '',
        'OKane2022·SPMe, 25°C 고정, 부분 가역 리튬 도금·SEI 저항·두 기공률 변화, 격자 ×4를 사용한다. SOC 20→80% 충전 중 4.2 V에 먼저 닿으면 CV로 전환한다. 5 A 방전과 충·방전 후 각 5분 휴지를 적용한다.', '',
        'SOH는 복제 상태의 기준시험 Qref/Q₀로 계산한다. 기준시험은 본 궤적과 EFC에 반영하지 않는다. 본 계산은 1초 간격 관측값과 사건 끝점을 저장하며, 기준시험 출력 간격은 기존 Experiment 기본값이다.', '',
        '## Linux 사전 검증', '',
        f"- Linux 통합 검증: {'통과' if batch.get('linux_smoke',{}).get('all_pass') else '미완료 또는 실패'}",
        '- 새 프로세스 체크포인트 복원, 방전 중 복제 진단 후 방전 지속, Parquet 재적분을 확인한다. Windows에서 통과한 결과를 Linux의 통과 근거로 대신하지 않는다.', '',
        '## 동시 실행 수 비교', '',
        '| 동시 실행 | 동일 4조건·각 0.05 EFC 벽시계 시간(초) | 부모·자식 합산 최대 RSS(MB) | EFC/시간 |',
        '|---:|---:|---:|---:|']
    for b in batch.get('benchmarks',[]):
        lines.append(f"| {b['workers']} | {b['wall_s']:.1f} | {b['peak_process_tree_rss_mb']:.1f} | {b['completed_efc_per_hour']:.3f} |")
    lines += ['', '4개 동시 실행이 2개보다 최소 5% 빠르고 최대 프로세스 RSS가 RAM의 75% 미만일 때 4개를 선택한다. 그 외에는 2개를 사용한다. 이 짧은 시험만으로 100 EFC 메모리 안정성이 확정되지는 않는다.', '',
        '## 전류별 실측 결과', '',
        '| 전류(A) | 도달 EFC | 마지막 SOH(%) | 충전시간 최소–최대(분) | 60분 초과 횟수 | 작업자 최대 RSS(MB) |',
        '|---:|---:|---:|---:|---:|---:|']
    for r in compact.get('results',[]):
        soh=r['diagnostics'][-1]['SOH_percent'] if r['diagnostics'] else None
        sohtext=f'{soh:.6f}' if soh is not None else '미측정'
        times=(f"{r['charge_min_minutes']:.3f}–{r['charge_max_minutes']:.3f}"
               if r['charge_min_minutes'] is not None else '미측정')
        rss=f"{r['worker_peak_rss_mb']:.1f}" if r['worker_peak_rss_mb'] is not None else '미측정'
        lines.append(f"| {r['current_A']:.1f} | {r['efc']:.6f} | {sohtext} | {times} | {r['time_limit_violations']} | {rss} |")
    if not compact.get('results'):
        lines.append('| — | 본 실험 결과 없음 | — | — | — | — |')
    for r in compact.get('results',[]):
        lines += ['', f"### {r['current_A']:.1f} A 기준시험", '', '| EFC | Qref(Ah) | SOH(%) |', '|---:|---:|---:|']
        for d in r['diagnostics']:
            lines.append(f"| {d['efc']:.6f} | {d['Qref_Ah']:.8f} | {d['SOH_percent']:.6f} |")
    if batch.get('error'):
        lines += ['', '## 중단 원인', '', batch['error']]
    lines += ['', '## 해석 범위와 저장 위치', '',
        '- 60분을 넘긴 조건도 수치적으로 유효하면 계산을 이어가며, 시간 제약을 만족하는 추천 후보에서는 제외한다.',
        '- 첫 100 EFC 결과만으로 수명 종료 EFC나 총수명 연장을 확정하지 않는다. LSTM·Elastic Net·베이지안 최적화는 이 실행에 포함하지 않았다.',
        '- 초기 SOH가 100%를 조금 넘거나 미세하게 오르는 값은 그대로 보존한다. 이를 성능 향상으로 해석하거나 100%로 잘라 처리하지 않는다.',
        '- 전체 배치·시계열·진단·체크포인트는 이 보고서와 같은 EBS 볼륨에 남는다. 자동 외부 업로드는 하지 않는다. EC2 종료 시 해당 EBS 보존 설정은 실행 전에 별도로 확인해야 한다.',
        '- 콘솔에는 축약 JSON을 출력한다. 전체 Parquet와 체크포인트 회수에는 보존 EBS 연결 등 별도의 회수 작업이 필요하다.', '',
        '## 설치 버전', '']
    lines.extend(f'- {name}: `{version}`' for name,version in batch['versions'].items())
    Path(out/'experiment_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
