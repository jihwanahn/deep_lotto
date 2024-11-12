import requests
from tqdm import tqdm
import random
import pandas as pd


class LottoRecommendationSystem:
    def __init__(self):
        self.past_results = []  # 과거 당첨 번호 저장
        self.weighted_criteria = {
            "sum": 0.4,
            "high_low": 0.2,
            "odd_even": 0.2,
            "previous_overlap": 0.1,
            "prime_count": 0.05,
            "multiple_of_three": 0.05,
        }

    def add_past_results(self, minDrwNo, maxDrwNo):
        """
        로또 API를 통해 과거 당첨 번호를 가져오고 시스템에 추가합니다.
        :param minDrwNo: 최소 회차 번호
        :param maxDrwNo: 최대 회차 번호
        """
        for i in tqdm(range(minDrwNo, maxDrwNo + 1)):
            try:
                req_url = f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={i}"
                response = requests.get(req_url)
                response.raise_for_status()
                lotto_info = response.json()

                # 각 회차의 당첨 번호 가져오기
                combination = [
                    lotto_info["drwtNo1"],
                    lotto_info["drwtNo2"],
                    lotto_info["drwtNo3"],
                    lotto_info["drwtNo4"],
                    lotto_info["drwtNo5"],
                    lotto_info["drwtNo6"],
                ]
                self.past_results.append(combination)

            except requests.RequestException as e:
                print(f"Error fetching data for draw number {i}: {e}")

    def calculate_statistics(self):
        """과거 당첨 번호에 대한 통계 계산"""
        stats = {
            "sum": {},
            "high_low": {},
            "odd_even": {},
            "prime_count": {},
            "multiple_of_three": {},
        }

        for result in self.past_results:
            result_sum = sum(result)
            stats["sum"][result_sum] = stats["sum"].get(result_sum, 0) + 1

            high = sum(1 for x in result if x > 23)
            stats["high_low"][high] = stats["high_low"].get(high, 0) + 1

            odd = sum(1 for x in result if x % 2 == 1)
            stats["odd_even"][odd] = stats["odd_even"].get(odd, 0) + 1

            prime_count = sum(1 for x in result if self.is_prime(x))
            stats["prime_count"][prime_count] = (
                stats["prime_count"].get(prime_count, 0) + 1
            )

            multiple_of_three_count = sum(1 for x in result if x % 3 == 0)
            stats["multiple_of_three"][multiple_of_three_count] = (
                stats["multiple_of_three"].get(multiple_of_three_count, 0) + 1
            )

        return stats

    @staticmethod
    def is_prime(num):
        """소수인지 확인"""
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    def recommend_numbers(self, num_recommendations=5):
        """추천 번호 생성"""
        stats = self.calculate_statistics()
        recommendations = []

        for _ in range(num_recommendations):
            recommendation = random.sample(range(1, 46), 6)
            score = self.calculate_score(recommendation, stats)
            recommendations.append((recommendation, score))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations

    def calculate_score(self, numbers, stats):
        """추천 번호 점수 계산"""
        score = 0
        numbers_sum = sum(numbers)
        high = sum(1 for x in numbers if x > 23)
        odd = sum(1 for x in numbers if x % 2 == 1)
        prime_count = sum(1 for x in numbers if self.is_prime(x))
        multiple_of_three_count = sum(1 for x in numbers if x % 3 == 0)

        if numbers_sum in stats["sum"]:
            score += stats["sum"][numbers_sum] * self.weighted_criteria["sum"]
        if high in stats["high_low"]:
            score += stats["high_low"][high] * self.weighted_criteria["high_low"]
        if odd in stats["odd_even"]:
            score += stats["odd_even"][odd] * self.weighted_criteria["odd_even"]
        if prime_count in stats["prime_count"]:
            score += (
                stats["prime_count"][prime_count]
                * self.weighted_criteria["prime_count"]
            )
        if multiple_of_three_count in stats["multiple_of_three"]:
            score += (
                stats["multiple_of_three"][multiple_of_three_count]
                * self.weighted_criteria["multiple_of_three"]
            )

        return score


# 실행 예제
lotto_system = LottoRecommendationSystem()
lotto_system.add_past_results(
    minDrwNo=1, maxDrwNo=1145
)  # 1회부터 10회까지 데이터 가져오기

recommendations = lotto_system.recommend_numbers(
    num_recommendations=5
)  # 추천 번호 3개 생성
for idx, (numbers, score) in enumerate(recommendations, 1):
    print(f"추천 번호 {idx}: {numbers} (점수: {score:.2f})")
