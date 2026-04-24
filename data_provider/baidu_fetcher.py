# -*- coding: utf-8 -*-
"""
===================================
BaiduFetcher - 百度财经数据源
===================================

数据来源：百度财经 API (finance.pae.baidu.com)
特点：免费、无需 Token、接口简洁

API 说明：
- 日K线接口：quotation_kline_ab
- 分钟线接口：quotation_minute_ab
- 返回格式：JSON，其中 marketData 为 CSV 格式字符串
  （分号分隔行，逗号分隔字段）

防封禁策略：
1. 需先访问 gushitong.baidu.com 获取 Cookie（BAIDUID）
2. 每次请求前随机休眠
3. 随机 User-Agent 轮换
4. 请求超时保护
"""

import logging
import os
import random
import time
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import requests

from .base import BaseFetcher, DataFetchError, STANDARD_COLUMNS

logger = logging.getLogger(__name__)

# 百度财经 API 基础 URL
BAIDU_FINANCE_API_URL = "https://finance.pae.baidu.com/vapi/v1/getquotation"

# 股票通页面（用于获取 Cookie）
GUSHITONG_BASE_URL = "https://gushitong.baidu.com/stock/ab-{code}"

# User-Agent 池
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
]

# 默认请求超时（秒）
_DEFAULT_TIMEOUT = 15


def _safe_float(val: str) -> float:
    """安全转换字符串为浮点数，处理空值和 '--' 占位符及 '+' 前缀"""
    val = val.strip()
    if not val or val == '--':
        return 0.0
    return float(val)


class BaiduFetcher(BaseFetcher):
    """
    百度财经数据源实现

    优先级：1（与 AkshareFetcher 同级）
    数据来源：百度财经 API
    接口：finance.pae.baidu.com

    API 返回的 marketData 格式（日K线，每行 19 个字段）：
    字段顺序：timestamp, time, open, close, volume, high, low, amount,
              range(涨跌额), ratio(涨跌幅), turnoverratio, preClose,
              ma5avgprice, ma5volume, ma10avgprice, ma10volume,
              ma20avgprice, ma20volume, (可能有第19字段)

    注意：需要先访问 gushitong.baidu.com 获取 BAIDUID Cookie，
    否则 API 会返回 403 Forbidden。
    """

    name = "BaiduFetcher"
    priority = int(os.getenv("BAIDU_PRIORITY", "1"))

    def __init__(self, sleep_min: float = 0.5, sleep_max: float = 1.5):
        """
        初始化 BaiduFetcher

        Args:
            sleep_min: 最小休眠时间（秒）
            sleep_max: 最大休眠时间（秒）
        """
        self.sleep_min = sleep_min
        self.sleep_max = sleep_max
        self._last_request_time: Optional[float] = None
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': random.choice(USER_AGENTS),
        })
        self._session_cookies_ready = False

    def _ensure_cookies(self, stock_code: str) -> None:
        """
        确保已获取百度 Cookie（BAIDUID）

        访问 gushitong.baidu.com 的股票页面来获取必要的 Cookie，
        之后才能成功调用 K 线数据 API。
        """
        if self._session_cookies_ready:
            return
        try:
            page_url = GUSHITONG_BASE_URL.format(code=stock_code)
            resp = self._session.get(page_url, timeout=_DEFAULT_TIMEOUT)
            if resp.status_code == 200 and 'BAIDUID' in dict(self._session.cookies):
                self._session_cookies_ready = True
                logger.debug("[BaiduFetcher] Cookie 获取成功")
            else:
                logger.warning(f"[BaiduFetcher] Cookie 获取异常: status={resp.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"[BaiduFetcher] Cookie 获取失败: {e}")

    def _enforce_rate_limit(self) -> None:
        """速率限制：随机休眠"""
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            min_interval = random.uniform(self.sleep_min, self.sleep_max)
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从百度财经获取原始 K 线数据

        Args:
            stock_code: 股票代码
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'

        Returns:
            原始 DataFrame
        """
        # 确保有 Cookie
        self._ensure_cookies(stock_code)

        # 速率限制
        self._enforce_rate_limit()

        logger.info(f"[BaiduFetcher] 请求 K 线数据: stock_code={stock_code}, range={start_date}~{end_date}")

        # 根据代码判断 is_kc 参数
        is_kc = "1" if stock_code.startswith('3') else "0"

        # 百度 API 要求 finClientType=pc 出现两次，手动拼接 URL
        url = (
            f"{BAIDU_FINANCE_API_URL}?"
            f"srcid=5353&pointType=string&group=quotation_kline_ab"
            f"&code={stock_code}&market_type=ab&newFormat=1"
            f"&is_kc={is_kc}&ktype=day"
            f"&finClientType=pc&finClientType=pc"
        )

        headers = {
            'Referer': f'https://gushitong.baidu.com/stock/ab-{stock_code}',
        }

        try:
            response = self._session.get(
                url,
                headers=headers,
                timeout=_DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            # 如果是 403，可能是 Cookie 过期，重置以便下次重新获取
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 403:
                self._session_cookies_ready = False
            raise DataFetchError(f"[BaiduFetcher] 请求失败: {e}") from e
        except ValueError as e:
            raise DataFetchError(f"[BaiduFetcher] 响应 JSON 解析失败: {e}") from e

        # 解析 marketData
        try:
            result = data.get('Result', {})
            new_market_data = result.get('newMarketData', {})
            market_data_str = new_market_data.get('marketData', '')

            if not market_data_str:
                raise DataFetchError(f"[BaiduFetcher] {stock_code} 未返回 marketData")
        except (AttributeError, TypeError) as e:
            raise DataFetchError(f"[BaiduFetcher] {stock_code} 响应结构异常: {e}") from e

        # 解析 CSV 格式数据
        rows = self._parse_market_data(market_data_str)
        if not rows:
            raise DataFetchError(f"[BaiduFetcher] {stock_code} 解析后无有效数据")

        df = pd.DataFrame(rows)

        # 日期过滤
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]

        logger.info(f"[BaiduFetcher] {stock_code} 获取原始数据: {len(df)} 条")
        return df

    def _parse_market_data(self, market_data_str: str) -> List[Dict[str, Any]]:
        """
        解析百度财经 marketData CSV 字符串

        格式：分号分隔行，逗号分隔字段
        字段顺序（19个）：
        0: timestamp (秒级时间戳)
        1: time (日期字符串, 如 "2024-01-15")
        2: open (开盘价)
        3: close (收盘价)
        4: volume (成交量)
        5: high (最高价)
        6: low (最低价)
        7: amount (成交额)
        8: range (涨跌额, 带正负号如 "+0.10")
        9: ratio (涨跌幅, 带正负号如 "+0.95")
        10: turnoverratio (换手率)
        11: preClose (前收盘价)
        12-17: ma5/ma10/ma20 均价和成交量 (可能为 "--")

        日K线数据中日期字段不含空格（格式 "YYYY-MM-DD"），
        分钟线数据日期含空格（格式 "YYYY-MM-DD HH:MM"），
        通过空格过滤只保留日K线数据。

        Args:
            market_data_str: CSV 格式字符串

        Returns:
            解析后的字典列表
        """
        rows = []
        seen_dates = set()

        line_items = market_data_str.split(';')
        for item in line_items:
            item = item.strip()
            if not item:
                continue

            fields = item.split(',')
            if len(fields) < 10:
                continue

            try:
                # 提取日期（字段1）
                date_str = fields[1].strip()
                # 过滤：只要纯日期格式（不含空格，即日线数据）
                if ' ' in date_str:
                    continue

                # 标准化日期格式
                if '-' in date_str:
                    normalized_date = date_str
                elif len(date_str) == 8:
                    normalized_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                else:
                    continue

                # 去重：同一日期只保留第一条
                if normalized_date in seen_dates:
                    continue
                seen_dates.add(normalized_date)

                # 百度返回的涨跌幅可能带 "+" 前缀，如 "+0.95"
                row = {
                    'date': normalized_date,
                    'open': _safe_float(fields[2]),
                    'close': _safe_float(fields[3]),
                    'volume': _safe_float(fields[4]),
                    'high': _safe_float(fields[5]),
                    'low': _safe_float(fields[6]),
                    'amount': _safe_float(fields[7]),
                    'pct_chg': _safe_float(fields[9]),
                }
                rows.append(row)
            except (ValueError, IndexError) as e:
                logger.debug(f"[BaiduFetcher] 跳过异常数据行: {item[:50]}... ({e})")
                continue

        return rows

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化百度财经数据

        _parse_market_data 已将字段映射为标准列名，
        此处只需确保列顺序和完整性。

        标准列名：date, open, high, low, close, volume, amount, pct_chg
        """
        df = df.copy()

        # 确保所有标准列都存在
        for col in STANDARD_COLUMNS:
            if col not in df.columns:
                df[col] = 0

        # 只保留标准列
        df = df[STANDARD_COLUMNS]

        return df

if __name__ == "__main__":
    
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    fetcher = BaiduFetcher()
    
    # 测试普通股票
    print("=" * 50)
    print("测试普通股票数据获取")
    print("=" * 50)
    try:
        df = fetcher.get_daily_data('600519')  # 茅台
        print(f"[股票] 获取成功，共 {len(df)} 条数据")
        print(df.tail())
    except Exception as e:
        print(f"[股票] 获取失败: {e}")
    
    # 测试 ETF 基金
    print("\n" + "=" * 50)
    print("测试 ETF 基金数据获取")
    print("=" * 50)
    try:
        df = fetcher.get_daily_data('512400')  # 有色龙头ETF
        print(f"[ETF] 获取成功，共 {len(df)} 条数据")
        print(df.tail())
    except Exception as e:
        print(f"[ETF] 获取失败: {e}")
    
    # 测试 ETF 实时行情
    print("\n" + "=" * 50)
    print("测试 ETF 实时行情获取")
    print("=" * 50)
    try:
        quote = fetcher.get_realtime_quote('512880')  # 证券ETF
        if quote:
            print(f"[ETF实时] {quote.name}: 价格={quote.price}, 涨跌幅={quote.change_pct}%")
        else:
            print("[ETF实时] 未获取到数据")
    except Exception as e:
        print(f"[ETF实时] 获取失败: {e}")
    
    # 测试港股历史数据
    print("\n" + "=" * 50)
    print("测试港股历史数据获取")
    print("=" * 50)
    try:
        df = fetcher.get_daily_data('00700')  # 腾讯控股
        print(f"[港股] 获取成功，共 {len(df)} 条数据")
        print(df.tail())
    except Exception as e:
        print(f"[港股] 获取失败: {e}")
    
    # 测试港股实时行情
    print("\n" + "=" * 50)
    print("测试港股实时行情获取")
    print("=" * 50)
    try:
        quote = fetcher.get_realtime_quote('00700')  # 腾讯控股
        if quote:
            print(f"[港股实时] {quote.name}: 价格={quote.price}, 涨跌幅={quote.change_pct}%")
        else:
            print("[港股实时] 未获取到数据")
    except Exception as e:
        print(f"[港股实时] 获取失败: {e}")

    # 测试市场统计
    print("\n" + "=" * 50)
    print("Testing get_market_stats (akshare)")
    print("=" * 50)
    try:
        stats = fetcher.get_market_stats()
        if stats:
            print(f"Market Stats successfully computed:")
            print(f"Up: {stats['up_count']} (Limit Up: {stats['limit_up_count']})")
            print(f"Down: {stats['down_count']} (Limit Down: {stats['limit_down_count']})")
            print(f"Flat: {stats['flat_count']}")
            print(f"Total Amount: {stats['total_amount']:.2f} 亿 (Yi)")
        else:
            print("Failed to compute market stats.")
    except Exception as e:
        print(f"Failed to compute market stats: {e}")

    # 测试筹码分布数据
    print("\n" + "=" * 50)
    print("测试筹码分布数据获取")
    print("=" * 50)
    try:
        chip = fetcher.get_chip_distribution('600519')  # 茅台
    except Exception as e:
        print(f"[筹码分布] 获取失败: {e}")

    # 测试行业板块排名
    print("\n" + "=" * 50)
    print("测试行业板块排名获取")
    print("=" * 50)
    try:
        rankings = fetcher.get_sector_rankings(n=5)
        if rankings:
            top, bottom = rankings
            print("涨幅榜 Top 5:")
            for sector in top:
                print(f"{sector['name']}: {sector['change_pct']}%")
            print("\n跌幅榜 Top 5:")
            for sector in bottom:
                print(f"{sector['name']}: {sector['change_pct']}%")
        else:
            print("未获取到行业板块排名数据")
    except Exception as e:
        print(f"[行业板块排名] 获取失败: {e}")
