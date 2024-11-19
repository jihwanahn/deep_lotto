import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import requests
from tqdm import tqdm
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import torch
import random
import json
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR


class BayesianLottoRecommendationSystem:
    def __init__(self, data_file="lotto_data.json", num_models=5):
        self.past_results = []
        self.data_file = data_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_models = num_models  # 앙상블에 사용할 모델 수
        print(f"Using device: {self.device}")

    def load_local_data(self):
        """
        로컬에 저장된 데이터를 불러옵니다.
        """
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r") as file:
                    self.past_results = json.load(file)
                print(f"로컬 데이터 로드 완료: {len(self.past_results)-1} 회차")
            except (json.JSONDecodeError, IOError) as e:
                print(f"로컬 데이터 파일 읽기 오류: {e}. 데이터를 초기화합니다.")
                self.past_results = []
        else:
            print("로컬 데이터 파일이 없습니다. 새로 데이터를 가져옵니다.")

    def save_local_data(self):
        """
        데이터를 로컬 파일에 저장합니다.
        """
        with open(self.data_file, "w") as file:
            json.dump(self.past_results, file, indent=4)
        print(f"로컬 데이터 저장 완료: {len(self.past_results)} 회차")

    def fetch_latest_data(self):
        """
        최신 회차 데이터를 가져옵니다.
        """
        try:
            req_url = (
                "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo="
            )
            response = requests.get(req_url + "1")  # 초기 요청으로 기본 구조 확인
            response.raise_for_status()
            max_draw_number = response.json().get("drwNo")
            if not max_draw_number:
                print("최신 회차 정보를 가져올 수 없습니다.")
                return

            last_stored_draw = 0
            if self.past_results:
                last_stored_draw = max(result["drwNo"] for result in self.past_results)

            # 새로운 데이터를 가져와서 로컬에 추가
            if last_stored_draw < max_draw_number:
                print(
                    f"최신 데이터 가져오 시작: {last_stored_draw + 1} ~ {max_draw_number} 회차"
                )
                for draw_no in tqdm(range(last_stored_draw + 1, max_draw_number + 1)):
                    response = requests.get(req_url + str(draw_no))
                    response.raise_for_status()
                    lotto_info = response.json()
                    if lotto_info.get("returnValue") == "success":
                        self.past_results.append(
                            {
                                "drwNo": draw_no,
                                "numbers": [
                                    lotto_info.get("drwtNo1"),
                                    lotto_info.get("drwtNo2"),
                                    lotto_info.get("drwtNo3"),
                                    lotto_info.get("drwtNo4"),
                                    lotto_info.get("drwtNo5"),
                                    lotto_info.get("drwtNo6"),
                                ],
                            }
                        )
                    else:
                        print(f"회차 {draw_no} 데이터가 누락되었습니다.")
                self.save_local_data()
            else:
                print("최신 데이터가 이미 로컬에 저장되어 있습니다.")

            # **최소 데이터 확보**
            if len(self.past_results) < 2:
                print("데이터가 부족하여 추가 데이터를 가져옵니다.")
                for draw_no in range(1, 1147):  # 기본적으로 1~10회차 데이터를 가져옴
                    response = requests.get(req_url + str(draw_no))
                    response.raise_for_status()
                    lotto_info = response.json()
                    if lotto_info.get("returnValue") == "success":
                        self.past_results.append(
                            {
                                "drwNo": draw_no,
                                "numbers": [
                                    lotto_info.get("drwtNo1"),
                                    lotto_info.get("drwtNo2"),
                                    lotto_info.get("drwtNo3"),
                                    lotto_info.get("drwtNo4"),
                                    lotto_info.get("drwtNo5"),
                                    lotto_info.get("drwtNo6"),
                                ],
                            }
                        )
                self.save_local_data()
        except requests.RequestException as e:
            print(f"데이터 가져오기 중 오류 발생: {e}")

    def preprocess_data(self):
        """
        데이터를 학습에 사용할 수 있도록 전처리합니다.
        """
        if len(self.past_results) < 2:
            raise ValueError(
                "데이터가 부족합니다. 최소 2회차 이상의 데이터가 필요합니다."
            )

        X, y = [], []
        for result in self.past_results[:-1]:
            numbers = result["numbers"]
            features = [1 if i in numbers else 0 for i in range(1, 46)]
            features.append(sum(numbers))
            features.append(sum(1 for x in numbers if x % 2 == 1))
            features.append(sum(1 for x in numbers if x <= 22))
            X.append(features)

            next_result = self.past_results[self.past_results.index(result) + 1][
                "numbers"
            ]
            y_next = [1 if i in next_result else 0 for i in range(1, 46)]
            y.append(y_next)

        X = torch.tensor(X, dtype=torch.float32).to(self.device)  # GPU로 이동
        y = torch.tensor(y, dtype=torch.float32).to(self.device)  # GPU로 이동

        # 배치 크기의 배수로 데이터 조정
        batch_size = 32
        num_complete_batches = len(X) // batch_size * batch_size
        X = X[:num_complete_batches]
        y = y[:num_complete_batches]

        return X, y

    def model(self, X, y=None):
        num_features = X.shape[1]
        hidden_units = 256

        # Dropout layer 추가
        dropout_rate = 0.5
        w1 = pyro.sample(
            "w1",
            dist.Normal(
                torch.zeros(num_features, hidden_units, device=self.device), 0.1
            ).to_event(2),
        )
        b1 = pyro.sample(
            "b1",
            dist.Normal(torch.zeros(hidden_units, device=self.device), 0.1).to_event(1),
        )
        w2 = pyro.sample(
            "w2",
            dist.Normal(
                torch.zeros(hidden_units, 45, device=self.device), 0.1
            ).to_event(2),
        )
        b2 = pyro.sample(
            "b2", dist.Normal(torch.zeros(45, device=self.device), 0.1).to_event(1)
        )

        # 배치 처리를 위한 plate
        if y is not None:
            with pyro.plate("data", X.shape[0]) as ind:
                X_batch = X[ind]
                hidden = torch.relu(torch.matmul(X_batch, w1) + b1)
                hidden = torch.nn.functional.dropout(
                    hidden, p=dropout_rate, training=True
                )  # 드롭아웃 적용
                logits = torch.matmul(hidden, w2) + b2
                probs = torch.sigmoid(logits)
                pyro.sample("obs", dist.Bernoulli(probs).to_event(1), obs=y)
        else:
            hidden = torch.relu(torch.matmul(X, w1) + b1)
            hidden = torch.nn.functional.dropout(
                hidden, p=dropout_rate, training=False
            )  # 드롭아웃 적용
            logits = torch.matmul(hidden, w2) + b2
            probs = torch.sigmoid(logits)
            return probs

    def guide(self, X, y=None):
        num_features = X.shape[1]
        hidden_units = 256

        # Variational parameters
        w1_mean = pyro.param(
            "w1_mean", torch.randn(num_features, hidden_units, device=self.device)
        )
        w1_std = pyro.param(
            "w1_std",
            torch.ones(num_features, hidden_units, device=self.device),
            constraint=dist.constraints.positive,
        )
        w1 = pyro.sample("w1", dist.Normal(w1_mean, w1_std).to_event(2))

        b1_mean = pyro.param("b1_mean", torch.randn(hidden_units, device=self.device))
        b1_std = pyro.param(
            "b1_std",
            torch.ones(hidden_units, device=self.device),
            constraint=dist.constraints.positive,
        )
        b1 = pyro.sample("b1", dist.Normal(b1_mean, b1_std).to_event(1))

        w2_mean = pyro.param(
            "w2_mean", torch.randn(hidden_units, 45, device=self.device)
        )
        w2_std = pyro.param(
            "w2_std",
            torch.ones(hidden_units, 45, device=self.device),
            constraint=dist.constraints.positive,
        )
        w2 = pyro.sample("w2", dist.Normal(w2_mean, w2_std).to_event(2))

        b2_mean = pyro.param("b2_mean", torch.randn(45, device=self.device))
        b2_std = pyro.param(
            "b2_std",
            torch.ones(45, device=self.device),
            constraint=dist.constraints.positive,
        )
        b2 = pyro.sample("b2", dist.Normal(b2_mean, b2_std).to_event(1))

    def train(self, X, y, num_steps=2000, batch_size=32, validation_split=0.2):
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "입력 데이터(X)와 출력 데이터(y)의 샘플 수가 일치하지 않습니다."
            )

        X = X.float()
        y = y.float()

        # 데이터 분할
        split_index = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]

        initial_lr = 0.001
        best_loss = float("inf")
        patience = 20
        no_improve_count = 0
        min_improvement = 0.001
        loss_history = []

        # 여러 모델을 학습
        for model_idx in range(self.num_models):
            print(f"Training model {model_idx + 1}/{self.num_models}")
            optimizer = Adam({"lr": initial_lr, "betas": (0.9, 0.999)})
            svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())

            num_batches = (X_train.shape[0] + batch_size - 1) // batch_size

            for step in range(num_steps):
                epoch_loss = 0.0
                perm = torch.randperm(X_train.shape[0])
                X_shuffled = X_train[perm]
                y_shuffled = y_train[perm]

                for i in range(num_batches):
                    batch_start = i * batch_size
                    batch_end = min((i + 1) * batch_size, X_train.shape[0])
                    X_batch = X_shuffled[batch_start:batch_end]
                    y_batch = y_shuffled[batch_start:batch_end]

                    loss = svi.step(X_batch, y_batch)
                    epoch_loss += loss

                avg_loss = epoch_loss / num_batches
                loss_history.append(avg_loss)

                # 검증 데이터로 성능 평가
                with torch.no_grad():
                    val_loss = svi.evaluate_loss(X_val, y_val)
                    print(f"Model {model_idx + 1} - Validation Loss: {val_loss:.2f}")

                # Early stopping 체크
                relative_improvement = (
                    (best_loss - avg_loss) / best_loss
                    if best_loss != float("inf")
                    else float("inf")
                )
                if relative_improvement > min_improvement:
                    best_loss = avg_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count >= patience and step > 500:
                    print(f"Early stopping for model {model_idx + 1} at step {step}")
                    break

        # 학습 과정 시각화
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.title("Training Loss Over Time")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.grid(True)
        plt.show()

    def recommend_numbers(self, X):
        if X.shape[1] != self.preprocess_data()[0].shape[1]:
            raise ValueError("입력 데이터의 특성 수가 학습 데이터와 일치하지 않습니다.")

        num_samples = 100
        all_probs = torch.zeros(45, device=self.device)

        # 여러 모델의 예측 결과를 평균
        for model_idx in range(self.num_models):
            print(f"Predicting with model {model_idx + 1}/{self.num_models}")
            predictive = pyro.infer.Predictive(
                self.model, guide=self.guide, num_samples=num_samples
            )

            X = X.to(self.device)

            with torch.no_grad():
                samples = predictive(X)
                probs = torch.zeros(45, device=self.device)
                for sample_idx in range(num_samples):
                    w1 = samples["w1"][sample_idx]
                    b1 = samples["b1"][sample_idx]
                    w2 = samples["w2"][sample_idx]
                    b2 = samples["b2"][sample_idx]

                    hidden = torch.relu(torch.matmul(X, w1) + b1)
                    hidden = torch.nn.functional.dropout(
                        hidden, p=0.5, training=False
                    )  # 드롭아웃 적용
                    logits = torch.matmul(hidden, w2) + b2
                    probs += torch.sigmoid(logits)[0]

                all_probs += probs / num_samples  # 각 모델의 예측 결과를 평균

        all_probs /= self.num_models  # 모든 모델의 예측 결과 평균

        # 상위 확률을 가진 번호 선택
        random_numbers = random.sample(range(1, 46), 3)
        top_indices = all_probs.argsort()[-3:]
        learned_numbers = [i + 1 for i in top_indices]

        # 최종 번호 조합 생성
        all_numbers = sorted(set(learned_numbers + random_numbers))

        # 6개의 번호가 되도록 조정
        while len(all_numbers) < 6:
            new_number = random.randint(1, 45)
            if new_number not in all_numbers:
                all_numbers.append(new_number)
        all_numbers.sort()

        return all_numbers


# 실행 예제
if __name__ == "__main__":
    lotto_system = BayesianLottoRecommendationSystem()
    lotto_system.load_local_data()
    lotto_system.fetch_latest_data()

    X, y = lotto_system.preprocess_data()
    print(f"데이터 크기 확인: X.shape = {X.shape}, y.shape = {y.shape}")

    lotto_system.train(X, y, num_steps=2000, batch_size=32)

    latest_numbers = lotto_system.past_results[-1]["numbers"]
    input_features = [1 if i in latest_numbers else 0 for i in range(1, 46)]
    input_features.append(sum(latest_numbers))
    input_features.append(sum(1 for x in latest_numbers if x % 2 == 1))
    input_features.append(sum(1 for x in latest_numbers if x <= 22))
    new_input = torch.tensor([input_features], dtype=torch.float32).to(
        lotto_system.device
    )

    recommendations = lotto_system.recommend_numbers(new_input)
    print(f"추천 번호: {recommendations}")
