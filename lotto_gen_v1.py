import requests
from tqdm import tqdm
import random
import pandas as pd


class LottoRecommendationSystem:
    def __init__(self):
        self.past_results = []  # 과거 당첨 번호 저장
        self.weighted_criteria = {
            "sum": 0.3,
            "high_low": 0.15,
            "odd_even": 0.15,
            "previous_overlap": 0.15,  # 이전 당첨번호와의 부분 일치 고려
            "prime_count": 0.1,
            "multiple_of_three": 0.05,
            "number_spacing": 0.1,  # 번호 간격 고려
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

    def recommend_numbers(self, num_recommendations=5, iterations=1000):
        """추천 번호 생성"""
        stats = self.calculate_statistics()
        best_recommendations = []
        seen_combinations = set()  # 이미 생성된 조합을 추적
        
        for _ in range(iterations):
            recommendation = random.sample(range(1, 46), 6)
            recommendation.sort()  # 정렬하여 동일한 조합을 확인하기 쉽게 함
            combination_tuple = tuple(recommendation)  # set에 저장하기 위해 tuple로 변환
            
            # 이미 본 조합이면 건너뛰기
            if combination_tuple in seen_combinations:
                continue
                
            score = self.calculate_score(recommendation, stats)
            seen_combinations.add(combination_tuple)
            
            # 상위 추천 번호만 유지
            if len(best_recommendations) < num_recommendations:
                best_recommendations.append((recommendation, score))
                best_recommendations.sort(key=lambda x: x[1], reverse=True)
            elif score > best_recommendations[-1][1]:
                best_recommendations[-1] = (recommendation, score)
                best_recommendations.sort(key=lambda x: x[1], reverse=True)
            
            # 돌연변이: 가장 높은 점수의 조합을 기반으로 새로운 조합 생성
            if best_recommendations and random.random() < 0.3:
                best_numbers = best_recommendations[0][0]
                mutation_count = random.randint(1, 3)
                new_recommendation = list(best_numbers)
                
                for _ in range(mutation_count):
                    idx = random.randint(0, 5)
                    new_number = random.randint(1, 45)
                    attempts = 0
                    # 중복되지 않는 새로운 숫자를 찾을 때까지 시도
                    while new_number in new_recommendation and attempts < 45:
                        new_number = random.randint(1, 45)
                        attempts += 1
                    new_recommendation[idx] = new_number
                
                new_recommendation.sort()
                new_tuple = tuple(new_recommendation)
                
                # 새로운 조합이 중복되지 않은 경우에만 처리
                if new_tuple not in seen_combinations:
                    seen_combinations.add(new_tuple)
                    score = self.calculate_score(new_recommendation, stats)
                    if score > best_recommendations[-1][1]:
                        best_recommendations[-1] = (new_recommendation, score)
                        best_recommendations.sort(key=lambda x: x[1], reverse=True)

        return best_recommendations

    def calculate_score(self, numbers, stats):
        """추천 번호 점수 계산"""
        score = 0
        numbers_sum = sum(numbers)
        high = sum(1 for x in numbers if x > 23)
        odd = sum(1 for x in numbers if x % 2 == 1)
        prime_count = sum(1 for x in numbers if self.is_prime(x))
        multiple_of_three_count = sum(1 for x in numbers if x % 3 == 0)

        # 기본 통계 점수 계산
        if numbers_sum in stats["sum"]:
            score += stats["sum"][numbers_sum] * self.weighted_criteria["sum"]
        if high in stats["high_low"]:
            score += stats["high_low"][high] * self.weighted_criteria["high_low"]
        if odd in stats["odd_even"]:
            score += stats["odd_even"][odd] * self.weighted_criteria["odd_even"]
        if prime_count in stats["prime_count"]:
            score += stats["prime_count"][prime_count] * self.weighted_criteria["prime_count"]
        if multiple_of_three_count in stats["multiple_of_three"]:
            score += stats["multiple_of_three"][multiple_of_three_count] * self.weighted_criteria["multiple_of_three"]

        # 번호 간격 점수 계산
        gaps = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
        avg_gap = sum(gaps) / len(gaps)
        gap_score = sum(1 for gap in gaps if 2 <= gap <= 8)  # 적절한 간격 보상
        score += (gap_score / len(gaps)) * self.weighted_criteria["number_spacing"]

        # 이전 당첨번호와의 부분 일치 분석
        overlap_score = 0
        for past_result in self.past_results[-10:]:  # 최근 10회 당첨번호만 고려
            matches = len(set(numbers) & set(past_result))
            if matches == 3:  # 3개 일치는 적절
                overlap_score += 1
            elif matches > 3:  # 너무 많은 일치는 감점
                overlap_score -= 0.5
        score += (overlap_score / 10) * self.weighted_criteria["previous_overlap"]

        # 미세 조정을 위한 랜덤 노이즈 추가 (0.1% 이내)
        score += random.uniform(0, 0.001) * score

        return score


# 실행 예제
lotto_system = LottoRecommendationSystem()
lotto_system.add_past_results(
    minDrwNo=1, maxDrwNo=1145
)  # 1회부터 10회까지 데이터 가져오기

recommendations = lotto_system.recommend_numbers(num_recommendations=5, iterations=10000)
for idx, (numbers, score) in enumerate(recommendations, 1):
    print(f"추천 번호 {idx}: {numbers} (점수: {score:.2f})")
