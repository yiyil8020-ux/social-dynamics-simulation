#!/usr/bin/env python3
"""
社会动力学推演引擎 — 10,000人×10年蒙特卡洛模拟
模拟参数基于中国城市居民统计数据、行为经济学研究和宏观经济预测
"""

import numpy as np
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple
import time

np.random.seed(42)

# ============================================================
# 配置
# ============================================================
N_AGENTS = 10_000
N_YEARS = 10
YEAR_START = 2026

# 宏观环境参数（每年动态变化）
MACRO = {
    2026: {"gdp_growth": 0.048, "inflation": 0.021, "unemployment": 0.055, "market_return": 0.06, "ai_disruption": 0.05},
    2027: {"gdp_growth": 0.046, "inflation": 0.023, "unemployment": 0.058, "market_return": 0.05, "ai_disruption": 0.10},
    2028: {"gdp_growth": 0.043, "inflation": 0.025, "unemployment": 0.065, "market_return": 0.03, "ai_disruption": 0.25},
    2029: {"gdp_growth": 0.040, "inflation": 0.024, "unemployment": 0.072, "market_return": 0.04, "ai_disruption": 0.35},
    2030: {"gdp_growth": 0.042, "inflation": 0.022, "unemployment": 0.068, "market_return": 0.07, "ai_disruption": 0.30},
    2031: {"gdp_growth": 0.045, "inflation": 0.020, "unemployment": 0.060, "market_return": 0.08, "ai_disruption": 0.20},
    2032: {"gdp_growth": 0.047, "inflation": 0.021, "unemployment": 0.056, "market_return": 0.06, "ai_disruption": 0.15},
    2033: {"gdp_growth": 0.048, "inflation": 0.022, "unemployment": 0.053, "market_return": 0.07, "ai_disruption": 0.12},
    2034: {"gdp_growth": 0.046, "inflation": 0.023, "unemployment": 0.052, "market_return": 0.05, "ai_disruption": 0.10},
    2035: {"gdp_growth": 0.045, "inflation": 0.022, "unemployment": 0.050, "market_return": 0.06, "ai_disruption": 0.08},
}

# ============================================================
# 智能体初始化
# ============================================================
@dataclass
class Agent:
    id: int
    age: int
    gender: int  # 0=male, 1=female
    education: str  # "high_school", "college", "master_plus"
    income: float  # 月收入
    savings: float  # 储蓄
    debt: float  # 负债
    health: float  # 0-1, 1=完美健康
    stress_resilience: float  # 0-10
    delay_gratification: float  # 0-1, 延迟满足能力
    info_literacy: float  # 0-1, 信息筛选能力
    social_network_size: int  # 弱关系数量
    has_emergency_fund: bool
    monthly_invest: float  # 每月投资金额
    sleep_hours: float
    exercise_freq: float  # 每周运动次数
    social_media_hours: float  # 每天社交媒体时间
    learning_hours_daily: float  # 每天深度学习时间
    relationship_status: str  # "single", "married_compatible", "married_conflict", "single_parent"
    has_children: bool
    industry: str
    skill_diversity: int  # 可变现技能数量
    fatal_errors: int = 0
    major_events: list = field(default_factory=list)
    yearly_snapshots: list = field(default_factory=list)
    is_bankrupt: bool = False
    is_depressed: bool = False
    years_survived: int = 0
    total_income: float = 0.0
    happiness_score: float = 0.0

def init_agents(n: int) -> List[Agent]:
    agents = []
    for i in range(n):
        age = int(np.clip(np.random.normal(33, 8), 22, 55))
        gender = np.random.randint(0, 2)
        
        # 教育水平
        edu_roll = np.random.random()
        if edu_roll < 0.22:
            education = "high_school"
            edu_mult = 0.65
        elif edu_roll < 0.77:
            education = "college"
            edu_mult = 1.0
        else:
            education = "master_plus"
            edu_mult = 1.45
        
        # 收入（对数正态分布）
        base_income = np.random.lognormal(np.log(8500), 0.7) * edu_mult
        age_factor = 1.0 + max(0, (age - 22)) * 0.015  # 年龄经验加成
        income = np.clip(base_income * age_factor, 3000, 120000)
        
        # 负债
        debt_ratio = np.random.exponential(0.8)
        if np.random.random() < 0.45:  # 45%有房贷
            debt_ratio = np.clip(debt_ratio + np.random.uniform(1.5, 4.0), 0, 6)
        debt = income * 12 * debt_ratio
        
        # 健康
        health = np.clip(np.random.normal(0.8, 0.12) - (age - 30) * 0.003, 0.2, 1.0)
        
        # 性格特征
        stress = np.clip(np.random.normal(5.5, 2.0), 1, 10)
        delay_g = np.clip(np.random.normal(0.5, 0.2), 0.05, 0.95)
        info_lit = np.clip(np.random.normal(0.5, 0.2), 0.05, 0.95)
        
        # 社交
        social = max(0, int(np.random.normal(10, 6)))
        
        # 储蓄（初始）
        initial_savings_rate = np.clip(np.random.normal(0.15, 0.10), 0, 0.5)
        savings = max(0, income * 12 * initial_savings_rate * np.random.uniform(0.5, 3))
        
        # 消费习惯
        sleep = np.clip(np.random.normal(7.0, 1.0), 4.5, 10)
        exercise = np.clip(np.random.normal(2.0, 1.5), 0, 7)
        social_media = np.clip(np.random.exponential(1.5), 0.1, 8)
        learning = np.clip(np.random.exponential(0.3), 0, 2)
        
        # 关系状态
        rel_roll = np.random.random()
        if age < 25:
            rel_probs = [0.55, 0.20, 0.10, 0.15]
        elif age < 35:
            rel_probs = [0.25, 0.30, 0.30, 0.15]
        else:
            rel_probs = [0.20, 0.25, 0.40, 0.15]
        rel_cum = np.cumsum(rel_probs)
        rel_idx = np.searchsorted(rel_cum, rel_roll)
        rel_status = ["single", "married_compatible", "married_conflict", "single_parent"][rel_idx]
        
        has_children = (rel_status in ["married_compatible", "married_conflict", "single_parent"]) and (age > 27 or np.random.random() < 0.3)
        
        # 行业
        industries = ["tech", "finance", "education", "healthcare", "manufacturing", 
                       "retail", "government", "media", "construction", "service"]
        industry = np.random.choice(industries)
        
        # 技能多样性
        skill_div = max(1, int(np.random.normal(2, 1)))
        
        has_ef = savings > income * 3  # 至少3个月应急金
        monthly_inv = income * np.clip(np.random.normal(0.10, 0.08), 0, 0.4) if has_ef else 0
        
        agent = Agent(
            id=i, age=age, gender=gender, education=education,
            income=round(income, 2), savings=round(savings, 2), debt=round(debt, 2),
            health=round(health, 3), stress_resilience=round(stress, 2),
            delay_gratification=round(delay_g, 3), info_literacy=round(info_lit, 3),
            social_network_size=social, has_emergency_fund=has_ef,
            monthly_invest=round(monthly_inv, 2), sleep_hours=round(sleep, 1),
            exercise_freq=round(exercise, 1), social_media_hours=round(social_media, 1),
            learning_hours_daily=round(learning, 2), relationship_status=rel_status,
            has_children=has_children, industry=industry, skill_diversity=skill_div
        )
        agents.append(agent)
    return agents

# ============================================================
# 年度推演逻辑
# ============================================================
def simulate_year(agent: Agent, year: int, macro: dict) -> None:
    if agent.is_bankrupt:
        return
    
    agent.age += 1
    
    # --- 随机事件 ---
    events = []
    
    # 裁员概率（受行业、年龄、技能影响）
    layoff_base = macro["unemployment"] * 0.3
    ai_risk = macro["ai_disruption"] * (1 - agent.skill_diversity * 0.2) * (1 - agent.info_literacy * 0.3)
    layoff_prob = layoff_base + ai_risk * 0.15
    if agent.age > 45:
        layoff_prob *= 1.3
    if agent.industry in ["tech", "finance", "media"]:
        layoff_prob *= 1.2
    if agent.industry in ["government", "healthcare"]:
        layoff_prob *= 0.5
    
    got_laid_off = np.random.random() < layoff_prob
    if got_laid_off:
        months_unemployed = max(1, int(np.random.exponential(4) * (1 - agent.skill_diversity * 0.15)))
        income_loss = agent.income * months_unemployed * 0.6  # 假设失业期间拿到部分补偿
        agent.savings -= income_loss
        agent.stress_resilience -= np.random.uniform(0.3, 1.5)
        events.append(f"裁员：失业{months_unemployed}个月")
        agent.fatal_errors += 0  # 裁员不是自己的错误
    
    # 健康事件
    health_event_prob = 0.05 + (1 - agent.health) * 0.15 + max(0, agent.sleep_hours - 6) * (-0.01) + agent.exercise_freq * (-0.005)
    health_event_prob = np.clip(health_event_prob, 0.01, 0.35)
    if agent.age > 40:
        health_event_prob += 0.03
    
    health_event = np.random.random() < health_event_prob
    if health_event:
        severity = np.random.choice(["minor", "moderate", "severe"], p=[0.5, 0.35, 0.15])
        if severity == "minor":
            cost = np.random.uniform(500, 3000)
            agent.health -= np.random.uniform(0.01, 0.03)
        elif severity == "moderate":
            cost = np.random.uniform(5000, 30000)
            agent.health -= np.random.uniform(0.03, 0.08)
            if agent.exercise_freq < 1:
                agent.fatal_errors += 1  # 不运动导致的健康问题算错误
        else:
            cost = np.random.uniform(50000, 200000)
            agent.health -= np.random.uniform(0.08, 0.20)
            agent.income *= np.random.uniform(0.5, 0.85)  # 重大健康问题影响收入能力
            if not agent.has_emergency_fund:
                agent.fatal_errors += 1  # 没有应急金就遇到重大健康问题
        agent.savings -= cost
        events.append(f"健康事件({severity})：支出{cost:.0f}元")
    
    # --- 收入变化 ---
    income_growth = macro["gdp_growth"] * np.random.uniform(0.5, 2.0)
    if agent.skill_diversity >= 2:
        income_growth += 0.03  # 技能多样性加成
    if agent.learning_hours_daily > 0.3:
        income_growth += 0.02  # 持续学习加成
    if agent.delay_gratification > 0.6:
        income_growth += 0.01  # 延迟满足加成
    if got_laid_off:
        income_growth -= 0.10  # 失业后收入下降
    
    agent.income *= (1 + income_growth)
    agent.income = max(2500, agent.income)  # 最低保底
    agent.total_income += agent.income * 12
    
    # --- 消费决策 ---
    # 基础生活支出
    base_expense_ratio = 0.45  # 基础生活成本占收入比例
    
    # 生活方式膨胀检测
    lifestyle_inflation = 0
    if agent.delay_gratification < 0.4:
        lifestyle_inflation = income_growth * 0.8  # 低延迟满足的人，收入涨消费也涨
    elif agent.delay_gratification < 0.6:
        lifestyle_inflation = income_growth * 0.4
    else:
        lifestyle_inflation = income_growth * 0.15  # 高延迟满足的人，消费增长远低于收入增长
    
    expense_ratio = base_expense_ratio + lifestyle_inflation
    
    # 有孩子的额外支出
    if agent.has_children:
        expense_ratio += 0.12
    
    # 社交媒体/冲动消费
    if agent.social_media_hours > 3:
        expense_ratio += 0.05  # 社交媒体引发的冲动消费
        agent.fatal_errors += 0.3  # 累积错误
    
    # 关系状态对支出的影响
    if agent.relationship_status == "married_conflict":
        expense_ratio += 0.08  # 冲突关系增加情绪性消费
        agent.fatal_errors += 0.2
    elif agent.relationship_status == "married_compatible":
        expense_ratio -= 0.05  # 和谐关系降低消费
    
    annual_expense = agent.income * 12 * expense_ratio
    
    # 额外随机大额支出（概率事件）
    if np.random.random() < 0.15:  # 15%概率遇到大额支出
        big_expense = np.random.uniform(agent.income * 0.5, agent.income * 3)
        annual_expense += big_expense
        # 情绪性消费决策检测
        if agent.stress_resilience < 4 and np.random.random() < 0.5:
            agent.fatal_errors += 1  # 情绪性大额消费
    
    # --- 储蓄与投资 ---
    savings_rate = max(0, (agent.income * 12 - annual_expense) / (agent.income * 12))
    
    # 行为模式影响储蓄率
    if agent.delay_gratification > 0.7:
        savings_rate = max(savings_rate, 0.25)  # 高延迟满足者强制储蓄
    
    annual_savings = agent.income * 12 * savings_rate
    agent.savings += annual_savings
    
    # 投资决策
    if agent.savings > agent.income * 6:  # 有足够应急金才投资
        agent.has_emergency_fund = True
        invest_amount = agent.income * 12 * min(savings_rate * 0.6, 0.20)
        
        # 投资回报（含市场波动和恐慌卖出）
        market_return = macro["market_return"] * np.random.normal(1.0, 0.4)
        
        # 恐慌卖出检测
        if market_return < -0.15 and agent.stress_resilience < 5:
            market_return *= 1.5  # 低点卖出，亏损放大
            agent.fatal_errors += 1
            events.append("恐慌卖出：在市场低点清仓")
        elif market_return < -0.15 and agent.stress_resilience >= 7:
            market_return *= 0.7  # 高抗压者反而加仓
            events.append("逆向加仓：在市场低点增加投入")
        
        agent.monthly_invest = invest_amount / 12
        agent.savings += invest_amount * market_return
    else:
        agent.has_emergency_fund = False
        agent.monthly_invest = 0
        if agent.savings > 0 and np.random.random() < 0.3:
            # 没有应急金但想投资 → 致命错误
            agent.fatal_errors += 0.5
    
    # --- 债务处理 ---
    if agent.debt > 0:
        annual_payment = min(agent.debt * 0.08, agent.savings * 0.5)  # 每年还8%
        agent.debt -= annual_payment
        agent.savings -= annual_payment
        
        # 以贷养贷检测
        if agent.savings < 0:
            new_debt = abs(agent.savings) * 1.2  # 以贷养贷，利息增加
            agent.debt += new_debt
            agent.savings = 0
            agent.fatal_errors += 2  # 严重错误
            events.append("以贷养贷：债务恶性循环启动")
    
    # --- 健康维护决策 ---
    # 运动
    if agent.exercise_freq >= 2:
        agent.health = min(1.0, agent.health + 0.01)
    else:
        agent.health -= np.random.uniform(0.005, 0.02)
    
    # 睡眠
    if agent.sleep_hours < 6:
        agent.health -= 0.01
        # 睡眠不足影响决策质量 → 消费增加
        annual_expense *= 1.02
    
    # --- 社交网络演化 ---
    if agent.learning_hours_daily > 0.2:
        agent.social_network_size += np.random.randint(0, 3)  # 学习带来新社交
    if agent.social_network_size < 5:
        agent.social_network_size = max(0, agent.social_network_size + np.random.randint(-1, 1))
    
    # 弱关系带来的机会
    if agent.social_network_size >= 10 and np.random.random() < 0.15:
        opportunity_boost = np.random.uniform(0.02, 0.10) * agent.income
        agent.income += opportunity_boost
        events.append(f"弱关系机会：收入提升{opportunity_boost:.0f}元/月")
    
    # --- 技能发展 ---
    if agent.learning_hours_daily > 0.3:
        if np.random.random() < 0.2:  # 20%概率技能提升
            agent.skill_diversity = min(5, agent.skill_diversity + 1)
            events.append("技能升级：获得新的可变现技能")
    
    # --- 心理状态 ---
    # 幸福感计算
    happiness = 0.0
    happiness += min(1, agent.income / 20000) * 0.2  # 收入贡献
    happiness += agent.health * 0.25  # 健康贡献
    happiness += (1 - min(1, agent.debt / max(1, agent.income * 12 * 3))) * 0.15  # 低负债贡献
    happiness += min(1, agent.exercise_freq / 3) * 0.1  # 运动贡献
    happiness += min(1, agent.sleep_hours / 7.5) * 0.1  # 睡眠贡献
    if agent.relationship_status == "married_compatible":
        happiness += 0.12
    elif agent.relationship_status == "married_conflict":
        happiness -= 0.08
    happiness += min(1, agent.social_network_size / 15) * 0.08  # 社交贡献
    
    # 压力损耗
    stress_damage = max(0, (10 - agent.stress_resilience) * 0.01)
    happiness -= stress_damage
    
    agent.happiness_score = np.clip(happiness, 0, 1)
    
    # 抑郁检测
    if agent.happiness_score < 0.25 and agent.stress_resilience < 4:
        if np.random.random() < 0.3:
            agent.is_depressed = True
            agent.income *= 0.7  # 抑郁影响工作能力
            events.append("心理健康危机：确诊抑郁/焦虑")
    
    # --- 破产检测 ---
    if agent.savings < -agent.income * 6:  # 负资产超过半年收入
        agent.is_bankrupt = True
        events.append("破产：资产归零")
    
    # 压力恢复
    if agent.exercise_freq >= 2:
        agent.stress_resilience = min(10, agent.stress_resilience + 0.1)
    agent.stress_resilience = np.clip(agent.stress_resilience + np.random.normal(0, 0.2), 1, 10)
    
    # 自然衰老
    if agent.age > 40:
        agent.health -= 0.005
    
    agent.health = np.clip(agent.health, 0.05, 1.0)
    agent.years_survived += 1
    
    # 记录年度快照
    agent.yearly_snapshots.append({
        "year": year,
        "age": agent.age,
        "income": round(agent.income, 2),
        "savings": round(agent.savings, 2),
        "debt": round(agent.debt, 2),
        "health": round(agent.health, 3),
        "happiness": round(agent.happiness_score, 3),
        "events": events,
        "savings_rate": round(savings_rate, 3),
    })

# ============================================================
# 分类与分析
# ============================================================
def classify_agents(agents: List[Agent]) -> Dict:
    categories = {
        "A_wealth_free": [],    # 财富自由
        "B_happy": [],          # 高度幸福
        "C_surviving": [],      # 平稳生存
        "D_anxious": [],        # 持续焦虑
        "E_failed": [],         # 严重失败
    }
    
    for a in agents:
        if a.is_bankrupt:
            categories["E_failed"].append(a)
            continue
        
        # 财富自由标准：被动收入 > 支出的1.5倍
        passive_income = a.savings * 0.04  # 4%提取率
        annual_expense = a.income * 12 * 0.5  # 假设50%收入用于生活
        if passive_income > annual_expense * 1.5 and a.savings > 0:
            categories["A_wealth_free"].append(a)
            continue
        
        # 高度幸福标准
        if a.happiness_score >= 0.6 and a.health >= 0.6 and a.savings > 0 and not a.is_depressed:
            categories["B_happy"].append(a)
            continue
        
        # 严重失败
        if a.is_depressed or a.health < 0.3 or a.debt > a.income * 12 * 4:
            categories["E_failed"].append(a)
            continue
        
        # 持续焦虑
        if a.savings < a.income * 2 or a.debt > a.income * 12 * 2 or a.happiness_score < 0.35:
            categories["D_anxious"].append(a)
            continue
        
        # 平稳生存
        categories["C_surviving"].append(a)
    
    return categories

def extract_patterns(categories: Dict) -> Dict:
    """提取幸存者与失败者的行为模式差异"""
    winners = categories["A_wealth_free"] + categories["B_happy"]
    losers = categories["E_failed"]
    
    def avg_field(agent_list, field_name):
        if not agent_list:
            return 0
        vals = [getattr(a, field_name) for a in agent_list]
        return np.mean(vals)
    
    patterns = {
        "winners_vs_losers": {
            "delay_gratification": {
                "winners": round(avg_field(winners, "delay_gratification"), 3),
                "losers": round(avg_field(losers, "delay_gratification"), 3),
            },
            "exercise_freq": {
                "winners": round(avg_field(winners, "exercise_freq"), 2),
                "losers": round(avg_field(losers, "exercise_freq"), 2),
            },
            "sleep_hours": {
                "winners": round(avg_field(winners, "sleep_hours"), 2),
                "losers": round(avg_field(losers, "sleep_hours"), 2),
            },
            "social_media_hours": {
                "winners": round(avg_field(winners, "social_media_hours"), 2),
                "losers": round(avg_field(losers, "social_media_hours"), 2),
            },
            "learning_hours_daily": {
                "winners": round(avg_field(winners, "learning_hours_daily"), 2),
                "losers": round(avg_field(losers, "learning_hours_daily"), 2),
            },
            "social_network_size": {
                "winners": round(avg_field(winners, "social_network_size"), 1),
                "losers": round(avg_field(losers, "social_network_size"), 1),
            },
            "fatal_errors": {
                "winners": round(avg_field(winners, "fatal_errors"), 2),
                "losers": round(avg_field(losers, "fatal_errors"), 2),
            },
            "skill_diversity": {
                "winners": round(avg_field(winners, "skill_diversity"), 2),
                "losers": round(avg_field(losers, "skill_diversity"), 2),
            },
            "stress_resilience": {
                "winners": round(avg_field(winners, "stress_resilience"), 2),
                "losers": round(avg_field(losers, "stress_resilience"), 2),
            },
            "info_literacy": {
                "winners": round(avg_field(winners, "info_literacy"), 3),
                "losers": round(avg_field(losers, "info_literacy"), 3),
            },
        },
        "winner_count": len(winners),
        "loser_count": len(losers),
    }
    
    # 收入起点 vs 最终结果
    all_agents_flat = []
    for cat_list in categories.values():
        all_agents_flat.extend(cat_list)
    
    # 计算收入起点与最终财富的相关性
    start_incomes = []
    final_savings = []
    for a in all_agents_flat:
        if a.yearly_snapshots:
            start_incomes.append(a.yearly_snapshots[0]["income"])
            final_savings.append(a.savings)
    
    if len(start_incomes) > 10:
        correlation = np.corrcoef(start_incomes, final_savings)[0, 1]
        patterns["income_to_wealth_correlation"] = round(correlation, 3)
    
    return patterns

def extract_fatal_error_patterns(losers: list) -> dict:
    """从失败者数据中提取高频致命错误"""
    if not losers:
        return {}
    
    error_counts = {
        "lifestyle_inflation": 0,
        "no_emergency_fund_investing": 0,
        "debt_spiral": 0,
        "health_neglect": 0,
        "emotional_spending": 0,
        "panic_selling": 0,
        "no_skill_backup": 0,
        "social_media_impulse": 0,
    }
    
    for a in losers:
        # 通过行为模式推断错误类型
        if a.delay_gratification < 0.35:
            error_counts["lifestyle_inflation"] += 1
        if not a.has_emergency_fund and a.monthly_invest > 0:
            error_counts["no_emergency_fund_investing"] += 1
        if a.debt > a.income * 12 * 3:
            error_counts["debt_spiral"] += 1
        if a.health < 0.4 and a.exercise_freq < 1:
            error_counts["health_neglect"] += 1
        if a.stress_resilience < 4 and a.social_media_hours > 3:
            error_counts["emotional_spending"] += 1
        if a.skill_diversity <= 1:
            error_counts["no_skill_backup"] += 1
        if a.social_media_hours > 4:
            error_counts["social_media_impulse"] += 1
    
    # 转为百分比
    n = len(losers)
    return {k: round(v / n * 100, 1) for k, v in error_counts.items()}

def compute_nonlinear_curves(agents: List[Agent]) -> dict:
    """计算非线性增长曲线数据"""
    # 按行为习惯分组，比较10年收入曲线
    daily_learners = [a for a in agents if a.learning_hours_daily > 0.3 and not a.is_bankrupt]
    non_learners = [a for a in agents if a.learning_hours_daily <= 0.1 and not a.is_bankrupt]
    
    def yearly_avg_income(agent_list, year_idx):
        incomes = []
        for a in agent_list:
            if len(a.yearly_snapshots) > year_idx:
                incomes.append(a.yearly_snapshots[year_idx]["income"])
        return round(np.mean(incomes), 2) if incomes else 0
    
    curves = {
        "learner_vs_nonlearner_income": {
            "yearly_learner_avg": [yearly_avg_income(daily_learners, y) for y in range(N_YEARS)],
            "yearly_nonlearner_avg": [yearly_avg_income(non_learners, y) for y in range(N_YEARS)],
            "learner_count": len(daily_learners),
            "nonlearner_count": len(non_learners),
        }
    }
    
    # 运动者 vs 不运动者的医疗支出代理（健康值下降幅度）
    exercisers = [a for a in agents if a.exercise_freq >= 2 and not a.is_bankrupt]
    sedentary = [a for a in agents if a.exercise_freq < 0.5 and not a.is_bankrupt]
    
    def yearly_avg_health(agent_list, year_idx):
        vals = []
        for a in agent_list:
            if len(a.yearly_snapshots) > year_idx:
                vals.append(a.yearly_snapshots[year_idx]["health"])
        return round(np.mean(vals), 3) if vals else 0
    
    curves["exerciser_vs_sedentary_health"] = {
        "yearly_exerciser_avg": [yearly_avg_health(exercisers, y) for y in range(N_YEARS)],
        "yearly_sedentary_avg": [yearly_avg_health(sedentary, y) for y in range(N_YEARS)],
        "exerciser_count": len(exercisers),
        "sedentary_count": len(sedentary),
    }
    
    # 高储蓄率 vs 低储蓄率的资产曲线
    high_savers = [a for a in agents if not a.is_bankrupt and a.yearly_snapshots and 
                   np.mean([s["savings_rate"] for s in a.yearly_snapshots]) > 0.25]
    low_savers = [a for a in agents if not a.is_bankrupt and a.yearly_snapshots and 
                  np.mean([s["savings_rate"] for s in a.yearly_snapshots]) < 0.10]
    
    def yearly_avg_savings(agent_list, year_idx):
        vals = []
        for a in agent_list:
            if len(a.yearly_snapshots) > year_idx:
                vals.append(a.yearly_snapshots[year_idx]["savings"])
        return round(np.mean(vals), 2) if vals else 0
    
    curves["high_vs_low_saver_assets"] = {
        "yearly_high_saver_avg": [yearly_avg_savings(high_savers, y) for y in range(N_YEARS)],
        "yearly_low_saver_avg": [yearly_avg_savings(low_savers, y) for y in range(N_YEARS)],
        "high_saver_count": len(high_savers),
        "low_saver_count": len(low_savers),
    }
    
    return curves

# ============================================================
# 主推演循环
# ============================================================
def run_simulation():
    print("=" * 60)
    print("社会动力学推演引擎 v3.0")
    print(f"正在初始化 {N_AGENTS:,} 个虚拟城市居民...")
    print("=" * 60)
    
    t0 = time.time()
    agents = init_agents(N_AGENTS)
    t1 = time.time()
    print(f"初始化完成 ({t1-t0:.2f}秒)")
    
    total_decisions = 0
    for year in range(YEAR_START, YEAR_START + N_YEARS):
        macro = MACRO[year]
        year_start = time.time()
        
        for agent in agents:
            simulate_year(agent, year, macro)
            # 每个agent每年约20个决策节点
            if not agent.is_bankrupt:
                total_decisions += 20
        
        year_time = time.time() - year_start
        alive = sum(1 for a in agents if not a.is_bankrupt)
        print(f"  {year}年 推演完成 | 存活: {alive:,} | 耗时: {year_time:.2f}s")
    
    total_time = time.time() - t0
    print(f"\n总推演完成 | 总决策模拟: ~{total_decisions:,}次 | 总耗时: {total_time:.2f}秒")
    
    # 分析
    print("\n正在分析结果...")
    categories = classify_agents(agents)
    patterns = extract_patterns(categories)
    fatal_errors = extract_fatal_error_patterns(categories["E_failed"])
    curves = compute_nonlinear_curves(agents)
    
    # 输出结果
    result = {
        "config": {
            "n_agents": N_AGENTS,
            "n_years": N_YEARS,
            "year_range": f"{YEAR_START}-{YEAR_START + N_YEARS - 1}",
            "total_decisions_simulated": total_decisions,
            "runtime_seconds": round(total_time, 2),
        },
        "category_counts": {
            "A_wealth_free": len(categories["A_wealth_free"]),
            "B_happy": len(categories["B_happy"]),
            "C_surviving": len(categories["C_surviving"]),
            "D_anxious": len(categories["D_anxious"]),
            "E_failed": len(categories["E_failed"]),
        },
        "category_percentages": {
            k: round(len(v) / N_AGENTS * 100, 2) 
            for k, v in categories.items()
        },
        "winner_loser_patterns": patterns,
        "fatal_error_patterns_percent": fatal_errors,
        "nonlinear_curves": curves,
    }
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("推演结果摘要")
    print("=" * 60)
    print(f"\n总决策模拟: ~{total_decisions:,} 次")
    print(f"\n人群分类:")
    for k, v in result["category_percentages"].items():
        count = result["category_counts"][k]
        print(f"  {k}: {count:,}人 ({v}%)")
    
    print(f"\n幸存者 vs 失败者行为差异:")
    p = patterns["winners_vs_losers"]
    for metric, vals in p.items():
        print(f"  {metric}: 赢家={vals['winners']} | 输家={vals['losers']}")
    
    if "income_to_wealth_correlation" in patterns:
        print(f"\n收入起点与最终财富的相关系数: {patterns['income_to_wealth_correlation']}")
    
    print(f"\n失败者高频致命错误 (% of losers):")
    for err, pct in fatal_errors.items():
        print(f"  {err}: {pct}%")
    
    # 保存完整结果
    with open("/Users/yiyi/无用的想法/社会动力学推演/simulation_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n完整结果已保存到 simulation_results.json")
    
    return result

if __name__ == "__main__":
    run_simulation()
