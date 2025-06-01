import streamlit as st
import pandas as pd
import numpy as np
import math
import ast
from datetime import datetime, timedelta
import statsmodels.stats.multitest
import scipy.stats as stats
import scipy.special as special
from statsmodels.stats.power import TTestIndPower
from dotenv import dotenv_values

from metabase import Mb_Client



secrets: dict = dotenv_values(".env")

MB_CLIENT: Mb_Client = Mb_Client(
    url=f"{secrets['mb_url']}",
    username=secrets["username"],
    password=secrets["password"]
)
    

def estimate_days(users_arr, sample_size):
    users_arr = ast.literal_eval(users_arr)
    users_arr = np.array(users_arr)
    avg_per_day = users_arr.mean()

    estimated_days = math.ceil(sample_size / avg_per_day)

    return avg_per_day, estimated_days


def calc_stats(mean_0, mean_1, var_0, var_1, len_0, len_1, alpha=0.05, required_power=0.8, pvalue=None, calc_mean=False):
    std = np.sqrt(var_0 / len_0 + var_1 / len_1)
    mean_abs = abs(mean_1 - mean_0)
    mean = mean_1 - mean_0
    sd = np.sqrt((var_0 * len_0 + var_1 * len_1) / 
                 (len_0 + len_1 - 2))
    if pvalue is None:
        pvalue = stats.norm.cdf(x=0, loc=mean_abs, scale=std) * 2
    elif calc_mean == False:
        std_corrected = np.abs(special.nrdtrisd(0, pvalue / 2, mean_abs))
        sd *= 1 + (std_corrected - std) / std
        std = std_corrected
    else:
        mean_abs = special.nrdtrimn(pvalue / 2, std, 0)
        mean = mean_abs
        if mean_0 > mean_1:
            mean *= -1

    cohen_d = mean_abs / sd
    bound_value = special.nrdtrimn(alpha / 2, std, 0)
    power = 1 - (stats.norm.cdf(x=bound_value, loc=mean_abs, scale=std) - 
                 stats.norm.cdf(x=-bound_value, loc=mean_abs, scale=std))
    analysis = TTestIndPower()
    # todo: добавить обработчик для нуля
    sample_size = analysis.solve_power(cohen_d, power=required_power, 
                                  nobs1=None, alpha=alpha)

    return {"pvalue": pvalue, "power": power, 
            "cohen_d": cohen_d, "sample_size": np.ceil(sample_size), 
            "enough": sample_size <= min(len_0, len_1),
            "ci": [np.array([stats.norm.ppf(alpha / 2, mean, std), 
                   stats.norm.ppf(1 - alpha / 2, mean, std)])]}


def generate_experiment_sql(
    platforms,
    pro_rights=None,
    edu_rights=None,
    sing_rights=None,
    practice_rights=None,
    books_rights=None,
    start_date=None,
    end_date=None,
    activation_event=None,
    event_params=None,
    exclude_event_params=False,
    subscription_sources=None,
    exclude_sources=False,
    metric="conversion to access"
):
    # --- Дата по умолчанию ---
    if not start_date or not end_date:
        end_date = datetime.today().date() - timedelta(14)
        start_date = end_date - timedelta(13)

    date_filter = f"date BETWEEN toDate('{start_date}') AND toDate('{end_date}')"

    # --- Таблица по платформе ---
    table_map = {
        "UG_WEB": "default.ug_rt_events_web"
    }

    # --- Подготовка условий прав ---
    def rights_conditions(label, rights, divisor=1):
        mapping = {
            "trial": 1,
            "paid subscription": 2,
            "lifetime": 3,
            "expired trial": 4,
            "expired paid subscription": 4
        }
        return [
            f"intDiv(rights, {int(divisor)}) % 10 = {mapping[r]}" for r in rights
        ]

    rights_filters = []

    if pro_rights:
        rights_filters.extend(rights_conditions("pro", pro_rights))
    if edu_rights:
        rights_filters.extend(rights_conditions("edu", edu_rights))
    if sing_rights:
        rights_filters.extend(rights_conditions("sing", sing_rights, 1e2))
    if practice_rights:
        rights_filters.extend(rights_conditions("practice", practice_rights, 1e3))
    if books_rights:
        rights_filters.extend(rights_conditions("books", books_rights, 1e4))

    rights_filter_sql = f"AND ({' OR '.join(rights_filters)})" if rights_filters else ""

    # --- Фильтр по событию ---
    event_filter = f"AND event = '{activation_event}'" if activation_event else ""

    # --- Фильтр по параметрам события ---
    if event_params:
        params = [f"value = '{p.strip()}'" for p in event_params.split(";") if p.strip()]
        joiner = " AND ".join if exclude_event_params else " OR ".join
        param_filter_sql = f"AND ({joiner(params)})"
    else:
        param_filter_sql = ""

    # --- Фильтр по источникам подписки ---
    if subscription_sources:
        sources = [f"funnel_source = '{s.strip()}'" for s in subscription_sources.split(";") if s.strip()]
        joiner = " AND ".join if exclude_sources else " OR ".join
        source_filter_sql = f"AND ({joiner(sources)})"
    else:
        source_filter_sql = ""

    # --- SQL шаблон ---
    union_queries = []
    for platform in platforms:
        table = table_map.get(platform, "default.ug_rt_events_app")
        source_filter = f"source = '{platform}'"

        if metric == "conversion to access":
            metric_sql = (
                "uniqExactIf(events.unified_id, sub_dt BETWEEN first_user_dt AND date_end)"
            )
        elif metric == "arpu":
            metric_sql = (
                "sumIf(revenue, sub_dt BETWEEN first_user_dt AND date_end "
                "AND ch_dt BETWEEN sub_dt AND sub_dt + INTERVAL 9 DAY)"
            )
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        platform_sql = f"""
            WITH
              toDate('{start_date}') AS date_start,
              toDate('{end_date}') AS date_end,

              subs AS (
                SELECT
                  *,
                  revenue_gross * CASE
                    WHEN lower(platform) LIKE '%ios%' THEN 0.7
                    WHEN lower(platform) LIKE '%and%' THEN 0.85
                    ELSE 1
                  END AS revenue
                FROM (
                  SELECT
                    subscription_id,
                    product_code,
                    minIf(datetime, event = 'Subscribed') AS sub_dt,
                    minIf(datetime, event = 'Charged') AS ch_dt,
                    minIf(datetime, event = 'Canceled') AS can_dt,
                    argMinIf(platform, datetime, event = 'Subscribed') AS platform,
                    argMinIf(trial, datetime, event = 'Subscribed') AS trial,
                    argMinIf(funnel_source, datetime, event = 'Subscribed') AS funnel_source,
                    argMinIf(product_id, datetime, event = 'Subscribed') AS product_id,
                    argMinIf(user_id, datetime, event = 'Subscribed') AS user_id,
                    argMinIf(unified_id, datetime, event = 'Subscribed') AS unified_id,
                    argMinIf(payment_account_id, datetime, event = 'Subscribed') AS payment_account_id,
                    argMinIf(usd_price, datetime, event = 'Charged') AS revenue_gross
                  FROM default.ug_subscriptions_events
                  WHERE date BETWEEN date_start AND date_end + INTERVAL 9 DAY
                    AND event IN ('Subscribed', 'Charged', 'Canceled')
                  GROUP BY subscription_id, product_code
                  HAVING toDate(sub_dt) BETWEEN date_start AND date_end
                  {source_filter_sql}
                )
              ),

              events AS (
                SELECT
                  unified_id,
                  source,
                  min(datetime) AS first_user_dt
                FROM {table}
                WHERE
                  {source_filter}
                  AND unified_id > 0
                  AND {date_filter}
                  {event_filter}
                  {param_filter_sql}
                  {rights_filter_sql}
                GROUP BY unified_id, source
              ),
              users_per_days AS (
                SELECT
                  source,
                  arraySort(x, y -> y, groupArray(users), groupArray(date)) AS users_arr
                FROM (
                  SELECT
                    source,
                    toDate(first_user_dt) AS date,
                    uniqExact(unified_id) AS users
                  FROM
                    events
                  GROUP BY
                    source, date
                )
                GROUP BY source
              ),

              user_stats AS (
                SELECT
                  unified_id,
                  source,
                  coalesce(uniqExactIf(events.unified_id, sub_dt BETWEEN first_user_dt AND date_end), 0) AS access_cnt,
                  coalesce(sumIf(subs.revenue, sub_dt BETWEEN first_user_dt AND date_end AND ch_dt BETWEEN sub_dt AND sub_dt + INTERVAL 9 DAY), 0) AS revenue
                FROM events
                LEFT JOIN subs
                ON events.unified_id = subs.unified_id
                WHERE
                  {source_filter}
                  AND unified_id > 0
                  {rights_filter_sql}
                GROUP BY unified_id, source
              )

            SELECT
              t.source AS source,
              date_end - date_start + 1 AS days,
              t.denominator AS denominator,
              t.numerator AS numerator,
              t.variance AS variance,
              arrayMap(x -> toUInt32(x), users_per_days.users_arr) AS users_arr
            FROM (
            SELECT
              source,
              uniqExact(user_stats.unified_id) AS denominator,
              {(
                "sum(access_cnt)"
                if metric == "conversion to access"
                else "sum(revenue)"
              )} AS numerator,
              {(
                "varSamp(user_stats.access_cnt)"
                if metric == "conversion to access"
                else "varSamp(user_stats.revenue)"
              )} AS variance
            FROM user_stats
            GROUP BY source
            ) as t
            INNER JOIN users_per_days
            ON t.source = users_per_days.source
        """
        union_queries.append(platform_sql.strip())

    return "\nUNION ALL\n".join(union_queries)




st.set_page_config(page_title="Sample Size Calculator for Experiments", layout="wide")

st.title("Sample Size Calculator for Experiments")

# --- Sidebar for experiment parameters ---
st.sidebar.header("Experiment Parameters")
num_branches = st.sidebar.number_input("Number of experiment branches:", min_value=1, step=1, value=2)
alpha = st.sidebar.number_input("Significance Level (alpha):", min_value=0.0001, max_value=1.0, value=0.05, step=0.01)
power = st.sidebar.number_input("Statistical Power:", min_value=0.0001, max_value=1.0, value=0.8, step=0.01)
expected_lift = st.sidebar.number_input("Expected Lift (%):", min_value=0.1, value=5.0, step=0.1)

# --- Two-column layout for filters ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Experiment Setup")
    # Metric
    metric = st.selectbox("Metric:", ["conversion to access", "arpu"])

    # Platform selection
    platforms = st.multiselect(
        "Select Platforms:",
        options=["UGT_IOS", "UG_IOS", "UGT_ANDROID", "UG_ANDROID", "UG_WEB"],
    )

    # Activation event
    activation_event = st.text_input("Activation Event (leave empty if not needed):")

    # Event parameters
    event_params = st.text_input("Event Parameters (separate by semicolon ';'):")
    exclude_event_params = st.checkbox("Exclude selected event parameters")

    # Subscription sources
    subscription_sources = st.text_input("Subscription Sources (separate by semicolon ';'):")
    exclude_sources = st.checkbox("Exclude selected sources")

with col2:
    # Date range
    st.subheader("Data Range and Rights")
    date_range = st.date_input(
        "Date Range (recommended: no later than two weeks ago):",
        help="Recommended to choose a range ending no later than 2 weeks ago",
        value=[]
    )

    # Rights filters
    def multiselect_rights(label):
        return st.multiselect(
            f"{label} Rights:",
            options=[
                "trial",
                "paid subscription",
                "lifetime",
                "expired trial",
                "expired paid subscription"
            ]
        )

    pro_rights = multiselect_rights("Pro")
    edu_rights = multiselect_rights("Edu")
    sing_rights = multiselect_rights("Sing")
    practice_rights = multiselect_rights("Practice")
    books_rights = multiselect_rights("Books")

# --- Calculate button and output ---
if st.button("Calculate"):
    sql_query = generate_experiment_sql(
        platforms=platforms,
        pro_rights=pro_rights,
        edu_rights=edu_rights,
        sing_rights=sing_rights,
        practice_rights=practice_rights,
        books_rights=books_rights,
        start_date=date_range[0] if date_range else None,
        end_date=date_range[1] if date_range else None,
        activation_event=activation_event,
        event_params=event_params,
        exclude_event_params=exclude_event_params,
        subscription_sources=subscription_sources,
        exclude_sources=exclude_sources,
        metric=metric
    )
    # print(sql_query)
    query_result = MB_CLIENT.post("dataset", sql_query)
    # print(query_result)
    table_results_df = pd.DataFrame({})
    for source in query_result["source"].unique():
        temp_df = query_result.loc[query_result["source"] == source]
        stat_res = calc_stats(
            mean_0=temp_df["numerator"].iloc[0] / temp_df["denominator"].iloc[0],
            mean_1=temp_df["numerator"].iloc[0] / temp_df["denominator"].iloc[0] * (1 + expected_lift / 100),
            var_0=temp_df["variance"].iloc[0],
            var_1=temp_df["variance"].iloc[0],
            len_0=temp_df["denominator"].iloc[0],
            len_1=temp_df["denominator"].iloc[0],
            alpha=alpha,
            required_power=power
        )
        # print(stat_res)
        # print(temp_df["users_arr"].iloc[0])
        # print(type(temp_df["users_arr"].iloc[0]))
        # print(stat_res["sample_size"] * num_branches)
        _, calc_days = estimate_days(temp_df["users_arr"].iloc[0], stat_res["sample_size"] * num_branches)
        # print(calc_days)
        if metric == "conversion to access":
            prefix = ""
            suffix = "%"
            metric_value = temp_df["numerator"].iloc[0] / temp_df["denominator"].iloc[0] * 100
        elif metric == "arpu":
            prefix = "$"
            suffix = ""
            metric_value = temp_df["numerator"].iloc[0] / temp_df["denominator"].iloc[0]
        table_results_df = pd.concat([
            table_results_df,
            pd.DataFrame({
                "Days": [calc_days],
                "Total Sample Size": [int(stat_res["sample_size"] * num_branches)],
                metric: [f"{prefix}{metric_value:.2f}{suffix}"],
                "Lift, %": [f"{expected_lift}%"],
                "Effect": f'{stat_res["cohen_d"]:.3f}'
                # "Expected CI": [f"[{stat_res['ci'][0][0]:.2%}; {stat_res['ci'][0][1]:.2%}]"]
            }, index=[source])
        ])
    
    # Dummy logic — replace with real calculation
    # days = 14
    # total_sample = 10000
    # effect = f"{expected_lift}%"
    # ci = "[-2.3%; +7.8%]"

    # result = pd.DataFrame({
    #     "Days": [days] * len(platforms),
    #     "Total Sample Size": [total_sample] * len(platforms),
    #     "Effect": [effect] * len(platforms),
    #     "Expected CI": [ci] * len(platforms),
    # }, index=platforms)

    st.markdown("### Result Table")
    # st.dataframe(result.style.format(precision=2), use_container_width=True)
    st.dataframe(table_results_df.style.format(precision=2), use_container_width=True)
    

