# BTC Hourly Regime Trader v2

نسخهٔ دوم پروژه، مدل EMA20/EMA50 را به‌طور کامل از هستهٔ تصمیم‌گیری حذف می‌کند. سیستم جدید روی کندل یک‌ساعته و ۱۸۰ روز تاریخچه کار می‌کند، اما به‌جای تقاطع دو میانگین ثابت، چهار **رویداد مستقل بازار** را تشخیص می‌دهد:

1. شکست کانال Donchian
2. خروج از فشردگی Bollinger
3. ادامهٔ روند پس از Pullback به KAMA
4. جهش جهت‌دار قیمت همراه افزایش حجم

KAMA و ADX فقط برای تشخیص رژیم بازار استفاده می‌شوند. مدل همیشه جهت `UP` یا `DOWN` می‌دهد، ولی موتور معامله فقط در صورت وجود یک رویداد جدید و عبور از همهٔ Fail-safeها اجازهٔ `LONG` یا `SHORT` می‌دهد؛ در غیر این صورت خروجی معاملاتی `WAIT` است.

## اصلاحات اصلی v2

- Dataset رویدادمحور با `event_id` یکتا
- هر رویداد فقط یک نمونهٔ مستقل
- ورود آموزشی روی `Open` کندل بعدی
- خروج در ۱، ۲ یا ۳ کندل بعد
- محاسبهٔ MFE، MAE و Triple Barrier محافظه‌کارانه
- مدل جهت + مدل احتمال قابل‌معامله‌بودن
- Probability calibration داخلی و گزارش Calibration Error
- Walk-forward دارای Gap برابر بیشترین افق
- Qualification جدا برای هر افق
- انتخاب افق بر اساس Edge، احتمال معامله‌پذیری و ثبات فولدها
- جلوگیری از بیش از یک معامله برای هر Event ID
- تفکیک کارمزد Maker، Taker، Slippage و Stress Cost
- اصلاح Resolve لایو بر اساس Open کندل بعدی و کسر هزینه فقط یک‌بار
- جلوگیری از Leakage خبر با `available_at = max(published_at, first_seen_at)`
- داشبورد مدیریتی جدید با همان ساختار قبلی

## اجرای پچ روی پروژه قبلی

1. از پوشهٔ `data` و `artifacts` نسخهٔ پشتیبان بگیر.
2. فایل‌های پچ را روی پروژه قبلی Copy/Replace کن.
3. `start_retrain.bat` را اجرا کن.
4. مدل قدیمی عمداً قابل استفاده نیست و پیام Retrain می‌دهد.
5. دیتابیس قبلی به‌صورت خودکار ستون‌های v2 را اضافه می‌کند.

## اجرای کامل از ابتدا

```bat
start_first_run.bat
```

## بازآموزی پس از نصب پچ

```bat
start_retrain.bat
```

## اجرای لایو و داشبورد

```bat
start_live.bat
```

## فایل‌های خروجی مهم

- `artifacts/models/latest.joblib`
- `artifacts/reports/latest_training_report.json`
- `artifacts/reports/latest_metrics.csv`
- `artifacts/reports/<model_id>_oof.csv`
- `data/runtime_state.json`
- `data/btc_ema_hourly.sqlite3`

## نکته ایمنی

این پروژه فقط Paper Trading است. ارسال سفارش واقعی پیاده‌سازی نشده و `paper_only: true` باید فعال بماند تا زمانی که حداقل یک افق، Qualification کامل خارج‌ازنمونه را پاس کند.

## Free scheduled deployment on GitHub

A stateless hourly GitHub Actions + GitHub Pages deployment is included in this package. See `GITHUB_FREE_DEPLOY_FA.md` for the exact Persian setup steps.
