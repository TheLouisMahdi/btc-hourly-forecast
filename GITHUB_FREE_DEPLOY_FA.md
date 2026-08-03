# اجرای کاملاً رایگان پروژه با GitHub Actions و GitHub Pages

این نسخه برای معماری یک‌ساعته پروژه آماده شده است:

- هر ساعت در دقیقه ۱۷ UTC یک اجرای موقت انجام می‌شود.
- ۱۸۰ روز کندل تازه دریافت می‌شود.
- اخبار تازه دریافت و یک پیش‌بینی ساخته می‌شود.
- خروجی در یک داشبورد استاتیک GitHub Pages منتشر می‌شود.
- مدل هر یکشنبه به‌صورت جداگانه دوباره آموزش داده می‌شود.
- هیچ سرور ۲۴ ساعته، SQLite دائمی یا Gradio دائمی لازم نیست.

## ۱. ساخت Repository

در GitHub یک Repository جدید بسازید:

- Visibility: **Public**
- گزینه‌های README و .gitignore را هنگام ساخت فعال نکنید.

GitHub Pages در پلن Free برای Repository عمومی قابل استفاده است. عمومی بودن یعنی کد، مدل و خروجی پیش‌بینی برای دیگران قابل مشاهده خواهد بود.

## ۲. آپلود پروژه

محتویات همین پوشه را در شاخه اصلی Repository، معمولاً `main`، قرار دهید. فایل‌های مهم زیر باید در GitHub دیده شوند:

```text
.github/workflows/hourly_forecast.yml
.github/workflows/weekly_retrain.yml
scripts/github_hourly_forecast.py
scripts/github_weekly_retrain.py
scripts/github_common.py
scripts/push_snapshot_branch.sh
requirements-github.txt
artifacts/models/latest.joblib
```

راه ساده با GitHub Desktop:

1. GitHub Desktop را نصب و وارد حساب شوید.
2. `File > Add local repository` را بزنید و پوشه پروژه را انتخاب کنید.
3. اگر پروژه Git نیست، `Create a repository` را بزنید.
4. یک Commit بسازید.
5. `Publish repository` را بزنید و Public بودن را انتخاب کنید.

## ۳. فعال‌کردن GitHub Pages

در Repository:

1. وارد `Settings` شوید.
2. از منوی چپ `Pages` را باز کنید.
3. در بخش `Build and deployment`، مقدار `Source` را روی **GitHub Actions** قرار دهید.

## ۴. بررسی اجازه اجرای Workflow

در Repository:

1. وارد `Settings > Actions > General` شوید.
2. در `Actions permissions` اجرای Actionها را فعال نگه دارید.
3. در `Workflow permissions` گزینه **Read and write permissions** را انتخاب کنید.
4. تنظیمات را ذخیره کنید.

Workflowها در خود فایل نیز مجوزهای لازم را درخواست می‌کنند، اما این تنظیم مانع خطای Push شاخه‌های snapshot می‌شود.

## ۵. اجرای اولین پیش‌بینی

1. وارد تب `Actions` شوید.
2. از سمت چپ `Hourly BTC forecast` را انتخاب کنید.
3. `Run workflow` را بزنید.
4. شاخه `main` را انتخاب و اجرا کنید.
5. بعد از موفق‌شدن Job، آدرس سایت در مرحله `Deploy GitHub Pages` و در `Settings > Pages` نمایش داده می‌شود.

آدرس معمولاً چنین شکلی دارد:

```text
https://USERNAME.github.io/REPOSITORY/
```

## ۶. اجرای اولین آموزش تازه

مدل موجود در پروژه فوراً قابل استفاده است. برای ساخت مدل تازه:

1. در تب `Actions`، Workflow با نام `Weekly BTC model retraining` را باز کنید.
2. `Run workflow` را بزنید.
3. پس از موفقیت، شاخه‌ای به نام `model-state` ساخته می‌شود.
4. اجرای ساعتی بعدی، مدل تازه را خودکار از این شاخه دریافت می‌کند.

اگر آموزش هفتگی شکست بخورد، مدل قبلی پاک نمی‌شود. پس از ۱۰ روز قدیمی‌شدن مدل، سیستم به‌صورت Fail-Safe تصمیم را مسدود می‌کند.

## ۷. زمان‌بندی‌ها

اجرای ساعتی:

```yaml
- cron: "17 * * * *"
```

یعنی هر ساعت، دقیقه ۱۷، بر اساس UTC. انتخاب دقیقه ۱۷ احتمال تأخیر ناشی از شلوغی ابتدای ساعت را کمتر می‌کند.

آموزش هفتگی:

```yaml
- cron: "47 3 * * 0"
```

یعنی یکشنبه ساعت 03:47 UTC.

## ۸. فایل‌ها و شاخه‌های خودکار

Workflowها دو شاخه snapshot می‌سازند:

- `forecast-state`: فقط `latest.json` و حداکثر ۳۰ روز تاریخچه ساعتی را نگه می‌دارد.
- `model-state`: فقط جدیدترین مدل و گزارش آموزش را نگه می‌دارد.

هر بار شاخه با `force push` بازنویسی می‌شود؛ بنابراین تاریخ Git با SQLite و مدل‌های تکراری بزرگ نمی‌شود.

## ۹. نکات مهم

- این داشبورد Gradio نیست؛ یک صفحه استاتیک همیشه در دسترس است.
- اجرای Python فقط چند دقیقه در هر ساعت فعال است.
- اجرای زمان‌بندی‌شده ممکن است چند دقیقه تأخیر داشته باشد.
- Workflow زمان‌بندی‌شده فقط از آخرین Commit شاخه پیش‌فرض اجرا می‌شود.
- GitHub ممکن است Workflow زمان‌بندی‌شده Repository عمومیِ بدون فعالیت را پس از ۶۰ روز غیرفعال کند. در آن حالت وارد Actions شوید و Workflow را دوباره Enable کنید.
- Repository عمومی روی Runner استاندارد هزینه دقیقه‌ای ندارد.
- مدل فعلی پروژه Qualification را پاس نکرده است؛ بنابراین حتی با اجرای صحیح، خروجی غالباً `WAIT` یا `BLOCKED` خواهد بود. این رفتار خطای Deploy نیست.
- این پروژه فقط برای پژوهش و Paper Trading است.
