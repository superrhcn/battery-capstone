# 배터리 디지털 트윈 · 100 EFC 예비실험

PyBaMM의 LG M50 가상 셀을 같은 조건으로 충방전해, 충전 전류에 따른 초기 열화와 계산 비용을 비교하는 프로젝트다. 오늘의 대상은 기준 셀 4번과 충전 전류 3.0·3.1·3.2·3.3 A다. 이 단계에서 LSTM이나 베이지안 최적화를 실행하지 않는다.

## 현재 상태

- 원본 프로젝트에서 소스·설정·결정 기록을 작업용 폴더로 복사했다. 원본 결과는 이 저장소에 포함하지 않는다.
- 이 파일 작성 시점에 GitHub 업로드, AWS 인스턴스 실행, 네 전류의 100 EFC 실험은 완료되지 않았다.
- `simulation/run_100efc.py`는 별도로 구현·검증 중인 실행기다. 파일이 생겼다는 사실만으로 검증 통과나 실험 완료를 의미하지 않는다.
- `docs/modeling_decisions.md`는 이전 결정 기록이다. 현재 실행 범위와 클라우드 준비 상태는 `docs/cloud_run_plan.md`를 함께 확인한다.

## 모듈과 데이터 구분

| 위치 | 입력 | 출력·역할 |
| --- | --- | --- |
| `configs/model.json` | 모델 옵션, 온도, SOC 범위, 전류, 전압 | 물리 계산 공통 설정 |
| `configs/cells_pilot9.json` | 입자 반지름 m, SEI 확산계수 m²/s | 셀 1~9의 파라미터 정의 |
| `configs/diagnostics.json` | 기준시험 전류·전압·휴지 | Q₀·Qref 측정 조건 |
| `simulation/battery.py` | 설정과 이전 물리 상태 | SPMe, CC–CV 제어, SOC 경계, 기준시험, R₁₀ |
| `simulation/verify_*.py` | 같은 환경의 검증 조건 | 수치 오차·상태 보존 검증 결과 |
| `simulation/run_pilot9.py` | 9개 셀과 선행 검증 결과 | 5 EFC 민감도 실험 |
| `outputs/` | 실행 중 생성 | 결과·로그·체크포인트. Git 추적 제외 |

SOC는 현재 사용 가능한 평형 용량을 기준으로 계산한다. SOH는 `100 × Qref/Q₀`, EFC는 `본 운전 누적 방전 Ah/Q₀`다. 기준시험을 위해 복제한 상태에서 사용한 전하량은 본 운전 EFC에 더하지 않는다.

## 물리 계산 환경

기준 로컬 환경은 Python 3.11.9다. 물리 계산만 준비할 때는 다음 파일을 사용한다.

```text
python -m pip install -r requirements-simulation.txt
```

`requirements.txt`에는 향후 학습용 PyTorch와 scikit-learn도 들어 있다. 기존 `simulation/check_env.py` 역시 이 두 패키지를 검사하므로, 물리 계산 전용 설치에서는 전체 검사 통과를 요구하지 않는다. 먼저 아래 버전 검사 후, 새 실행기의 단일 주기 시험과 재개 검증을 진행한다.

```text
python -c "import importlib.metadata as m; print({p:m.version(p) for p in ['pybamm','pybammsolvers','casadi','numpy','scipy','pandas','pyarrow']})"
```

Windows에서 확인한 버전과 Linux 클라우드에서 실제 설치·실행되는지는 별개의 검사다. 설치 오류가 나면 버전을 임의로 바꾸지 말고 Python 버전·운영체제·오류 메시지를 남긴다.

## 실험 조건

25°C, 초기 SOC 20%, 충전 종료 SOC 80%, 방전 5 A, 충전·방전 후 각각 5분 휴지를 적용한다. 충전 중 4.2 V에 먼저 닿으면 CV로 전환하며, SOC 80%에 먼저 닿으면 충전을 끝낸다. 기준 셀 4번은 음극 입자 반지름 5.86 μm, SEI 용매 확산계수 2.5×10⁻²² m²/s다.

100 EFC는 방전 전하량이 Q₀의 100배라는 뜻이며, 20→80% 운전 100회를 뜻하지 않는다. 충전 시간이 60분을 넘은 후보는 추천 제약 위반으로 기록하되, 물리 계산이 유효하면 열화 비교는 계속한다.

AWS 사양·동시 실행·중간 저장·통과 기준은 [클라우드 실행 계획](docs/cloud_run_plan.md)에 정리했다. 비밀번호, 인증키, 원시 결과를 소스 저장소에 넣지 않는다.
