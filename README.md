# 배터리 디지털 트윈 · 100 EFC 예비실험

PyBaMM의 LG M50 가상 셀을 같은 조건으로 충방전해, 충전 전류에 따른 초기 열화와 계산 비용을 비교하는 프로젝트다. 오늘의 대상은 기준 셀 4번과 충전 전류 3.0·3.1·3.2·3.3 A다. 이 단계에서 LSTM이나 베이지안 최적화를 실행하지 않는다.

## 현재 상태

- 원본 프로젝트에서 소스·설정·결정 기록을 작업용 폴더로 복사했다. 원본 결과는 이 저장소에 포함하지 않는다.
- 공개 저장소 [battery-capstone](https://github.com/superrhcn/battery-capstone)에 소스·설정·문서·스크립트 25개 파일을 게시하고 커밋 `777153d2f8b695fa12510dcd50ac5d3ec16249c8`의 반영을 확인했다. 이 README는 이후 진행 상황을 갱신한 문서다.
- `simulation/run_100efc.py`를 구현했다. 로컬 ×1에서 새 프로세스 재개·진단 보존·Parquet 왕복 검증을 통과했고, ×4에서 셀 4번·3.3 A·0.05 EFC 짧은 계산을 마쳤다. 상세 수치는 [실행기 검증 기록](docs/runner_validation_2026-10-28.md)에 있다.
- AWS `us-east-1`에 `m7i.2xlarge`(8 vCPU·32 GiB RAM, 암호화한 40 GiB gp3)를 2026-10-28 16:34 KST에 시작했다. 생성 시 확인한 Linux On-Demand 컴퓨팅 요금은 시간당 USD 0.4032이며 EBS·공인 IPv4·세금은 별도다.
- AWS Linux Python 3.11.17에서 `LINUX_SMOKE_PASS`를 확인했고, 동시 2건·4건 비교 모두 네 전류의 0.05 EFC 계산을 성공적으로 끝냈다. 이후 **본 100 EFC 네 전류를 병렬 실행 중이며, 최종 결과를 기다리고 있다.**
- 최근 시스템 로그에서 3.0·3.1·3.2·3.3 A 궤적은 각각 약 3.634 EFC까지 진행했고, 실행 프로세스 트리의 합산 RSS는 8,504.8 MiB(약 8.31 GiB)였다. 동시 실행 속도의 정확한 비교 비율과 100 EFC 최종 결과는 아직 보고하지 않는다.
- 사용자 예산은 약 한 달 600,000원이다. 2026-10-28~11-28 기간 USD 350 예산 알림을 설정했지만 강제 지출 상한은 아니다.
- `docs/modeling_decisions.md`, `docs/cloud_run_plan.md`와 검증 문서는 각 작성 시점의 기록이다. 이후 클라우드 진행 상태는 이 README의 현재 상태를 따른다.

## 모듈과 데이터 구분

| 위치 | 입력 | 출력·역할 |
| --- | --- | --- |
| `configs/model.json` | 모델 옵션, 온도, SOC 범위, 전류, 전압 | 물리 계산 공통 설정 |
| `configs/cells_pilot9.json` | 입자 반지름 m, SEI 확산계수 m²/s | 셀 1~9의 파라미터 정의 |
| `configs/diagnostics.json` | 기준시험 전류·전압·휴지 | Q₀·Qref 측정 조건 |
| `simulation/battery.py` | 설정과 이전 물리 상태 | SPMe, CC–CV 제어, SOC 경계, 기준시험, R₁₀ |
| `simulation/verify_*.py` | 같은 환경의 검증 조건 | 수치 오차·상태 보존 검증 결과 |
| `simulation/run_pilot9.py` | 9개 셀과 선행 검증 결과 | 5 EFC 민감도 실험 |
| `simulation/run_100efc.py` | 셀·전류·격자·EFC 종료·결과 경로 | 주기별 Parquet, 진단, 체크포인트, 요약 |
| `simulation/run_cloud_batch.py` | Linux 환경·배치 출력 경로 | 재검증, 동시 2·4건 비교, 네 전류 100 EFC 배치 관리 |
| `simulation/write_cloud_report.py` | 저장된 배치·궤적 결과 | 실제 결과에 근거한 실험 보고서 생성 |
| `scripts/cloud_bootstrap.sh` | 공개 저장소 URL·고정 커밋 | 환경 설치, 배치 실행, 종료 시 전원 끄기·독립 12시간 타이머 |
| `outputs/` | 실행 중 생성 | 결과·로그·체크포인트. Git 추적 제외 |

SOC는 현재 사용 가능한 평형 용량을 기준으로 계산한다. SOH는 `100 × Qref/Q₀`, EFC는 `본 운전 누적 방전 Ah/Q₀`다. 기준시험을 위해 복제한 상태에서 사용한 전하량은 본 운전 EFC에 더하지 않는다.

## 물리 계산 환경

기준 로컬 환경은 Windows Python 3.11.9이며, 실제 AWS 환경은 Linux Python 3.11.17이다. 패키지 고정 버전은 같고 Linux 짧은 계산 검사도 통과했다. 물리 계산만 준비할 때는 다음 파일을 사용한다.

```text
python -m pip install -r requirements-simulation.txt
```

`requirements.txt`에는 향후 학습용 PyTorch와 scikit-learn도 들어 있다. 기존 `simulation/check_env.py` 역시 이 두 패키지를 검사하므로, 물리 계산 전용 설치에서는 전체 검사 통과를 요구하지 않는다. 다른 실행 환경에서는 아래 버전 검사 후, 새 실행기의 짧은 계산과 재개 검증을 다시 진행한다.

```text
python -c "import importlib.metadata as m; print({p:m.version(p) for p in ['pybamm','pybammsolvers','casadi','numpy','scipy','pandas','pyarrow']})"
```

다른 OS·Python 패치 버전에서는 환경 검사와 짧은 계산을 다시 수행한다. 설치 오류가 나면 버전을 임의로 바꾸지 말고 Python 버전·운영체제·오류 메시지를 남긴다. 이번 Linux 설치·짧은 검사 통과는 장기 계산 완료를 뜻하지 않는다.

## 실험 조건

25°C, 초기 SOC 20%, 충전 종료 SOC 80%, 방전 5 A, 충전·방전 후 각각 5분 휴지를 적용한다. 충전 중 4.2 V에 먼저 닿으면 CV로 전환하며, SOC 80%에 먼저 닿으면 충전을 끝낸다. 기준 셀 4번은 음극 입자 반지름 5.86 μm, SEI 용매 확산계수 2.5×10⁻²² m²/s다.

100 EFC는 방전 전하량이 Q₀의 100배라는 뜻이며, 20→80% 운전 100회를 뜻하지 않는다. 충전 시간이 60분을 넘은 후보는 추천 제약 위반으로 기록하되, 물리 계산이 유효하면 열화 비교는 계속한다.

AWS 실험 설계·통과 기준은 [클라우드 실행 계획](docs/cloud_run_plan.md), 실제 배치 절차는 `simulation/run_cloud_batch.py`와 `scripts/cloud_bootstrap.sh`에 있다. 배치가 끝나거나 부트스트랩이 실패하면 전원을 끄고, 별도의 12시간 종료 타이머도 설정한다. 전원을 꺼도 남은 EBS 등에는 비용이 발생할 수 있다. 비밀번호, 인증키, 원시 결과를 소스 저장소에 넣지 않는다.

현재 본 운전 출력은 1초 간격에 이벤트 끝점·단계 경계의 중복 행을 포함한다. 0.1초 전환 구간 기록과 기준시험의 1초 출력은 아직 적용하지 않았다. 체크포인트는 이 실행기가 생성한 신뢰하는 파일만 읽고, 코드·설정·의존성 식별자가 같은 실행에서만 재개한다. Windows 결과를 Linux에서 이어 읽는 교차 환경 재개는 검증하지 않았다.
