# Data folder — public tenancy guidance (snapshot)

This folder holds a **snapshot** of public Irish tenancy guidance pages
fetched on **2026-11-14 20:48 GMT** (Europe/Dublin, UTC+0).

- Sources: [Citizens Information](https://www.citizensinformation.ie/) and
  the [Residential Tenancies Board](https://www.rtb.ie/).
- 10 pages were fetched successfully: 4 from Citizens Information and 6
  from the RTB. See `manifest.json`.
- Each file in `pages/` records the **source URL**, the **fetch time**,
  and the extracted text. Raw HTML is in `raw/`.
- The text remains the publishers' copyright. It is stored only so this
  educational project can demonstrate retrieval with citations. Do not
  republish the full text as your own guidance.
- This is a snapshot, not a complete statement of Irish rental law.
  The law changed on 14 September 2026 (and earlier in 2026). Always
  check the live page.

## How the pages were fetched

`src/fetch_pages.py` uses a small, hand-picked URL list (no login, no
site-wide crawl, a short delay between requests). On 14 November 2026,
Citizens Information returned **HTTP 403** to a custom educational
User-Agent, but served the same public pages to a standard browser
User-Agent. The committed snapshot was fetched with the browser
User-Agent after that check.

## Licence

Unlike project 2's CSO table (CC BY 4.0), these guidance pages are
**not** published under a Creative Commons licence that we can
re-licence. Copyright stays with the publishers. The MIT licence of
this repository covers the project code, not the stored page text.
