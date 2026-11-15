"""
Write a SYNTHETIC "client document pack" for the RAG demo.

Brightwater Facilities Ltd is an invented Galway facilities-management
company. Every policy, figure, name and address below is fictional and
exists only to give the retrieval system realistic internal documents to
search. Two documents are rendered as PDF, one as plain text and the rest
as Markdown, so the ingestion code has to handle mixed formats.

    python src/generate_client_docs.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "client_docs"
BANNER = "SYNTHETIC EXAMPLE DOCUMENT - Brightwater Facilities Ltd is fictional."

DOCS: dict[str, tuple[str, str]] = {}

DOCS["employee_handbook"] = ("md", """
# Employee Handbook

## Working hours
The standard working week is 39 hours, Monday to Friday. Core hours are 10:00 to 16:00; outside core hours staff may start between 07:30 and 10:00 and finish between 16:00 and 19:00 by agreement with their line manager. Site operatives follow the rota published every Thursday for the following week.

## Annual leave
Full-time employees receive 23 days of annual leave per year, plus Irish public holidays. After five years of continuous service this rises to 25 days. Leave requests are made in the HR portal at least two weeks in advance for periods longer than three days. Up to five unused days may be carried over into the first quarter of the next leave year.

## Sick leave
After passing probation, employees receive full pay for up to 10 days of certified sick leave in any rolling 12-month period, in addition to statutory sick pay. A medical certificate is required from the third consecutive day of absence. Employees must phone their line manager before 09:30 on the first day of absence; a text or email alone is not sufficient.

## Probation
New employees serve a six-month probation period. A review meeting takes place at month three and again before the end of month six. Probation may be extended once by up to three months.

## Remote working
After probation, office-based staff may work remotely for up to two days per week, agreed with their manager. Remote work must be carried out from within Ireland. Site-based roles are not eligible for remote working.

## Bereavement and compassionate leave
Employees may take up to five days of paid bereavement leave on the death of an immediate family member, and up to two days for an extended family member.

## Notice periods
During probation either side may end employment with one week's notice. After probation the notice period is four weeks, or the statutory minimum if that is longer.
""")

DOCS["expenses_policy"] = ("pdf", """
# Travel and Expenses Policy

## Submitting claims
All expense claims are submitted through the SpendDesk portal within 30 days of the expense being incurred. Claims submitted later than 30 days will only be paid with written approval from the Finance Manager. An itemised receipt is required for every expense over 10 euro.

## Approval limits
Line managers approve claims up to 500 euro. Claims above 500 euro also require approval from the Finance Manager. Claims are reimbursed in the next monthly payroll run after approval.

## Mileage
Use of a private car on company business is paid at 0.43 euro per kilometre for the first 1,500 km in a calendar year and 0.26 euro per kilometre after that. Commuting between home and your normal place of work is not claimable. Tolls and parking are reimbursed with receipts.

## Subsistence
When working more than 5 hours away from your normal place of work, a day allowance of 15 euro may be claimed; for more than 10 hours the allowance is 25 euro. Alcohol is never reimbursed.

## Accommodation
Overnight stays must be booked through the travel desk. The nightly cap is 140 euro in Dublin and 110 euro elsewhere in Ireland, including breakfast. Upgrades are at the employee's own cost.

## Company cards
Company credit cards are issued to managers only. Personal use of a company card is a disciplinary matter, even if repaid.
""")

DOCS["it_security_policy"] = ("md", """
# IT Security Policy

## Passwords and sign-in
Passwords must be at least 14 characters long. Multi-factor authentication (MFA) is mandatory for email, the HR portal and SpendDesk. Passwords are not changed on a fixed schedule; they must be changed immediately if a compromise is suspected. Password managers approved by IT may be used.

## Devices
Laptops must lock automatically after 5 minutes of inactivity. USB storage devices are blocked on all company laptops. A lost or stolen device must be reported to the IT helpdesk on extension 4400 within 2 hours of discovering the loss.

## Phishing and incidents
Suspicious emails must be reported using the Report Phishing button in Outlook, or by forwarding to security@brightwater.example. Do not click links or open attachments in a suspected phishing email. Security incidents are logged by IT and reviewed every Monday.

## Patching and backups
Critical security patches are installed within 72 hours of release; other patches within 14 days. File server backups run nightly at 01:00 and are retained for 35 days. A restore test is carried out every quarter.

## Acceptable use
Company systems may be used for limited personal use during breaks. Installing unapproved software, using personal cloud storage for company files, or sharing accounts is not permitted.
""")

DOCS["service_level_agreement"] = ("pdf", """
# Client Service Level Agreement (standard terms)

## Service hours
Standard service hours are 07:00 to 19:00, Monday to Friday, excluding public holidays. A 24/7 emergency line is available for Priority 1 incidents only.

## Priority definitions
Priority 1 (P1) means a fault that stops the client's site operating or creates a safety risk, for example a flood or a failed fire alarm panel. Priority 2 (P2) means a fault that significantly affects operations but has a workaround. Priority 3 (P3) covers routine requests and minor defects.

## Response and resolution targets
For P1 incidents Brightwater will respond within 1 hour and aim to resolve or make safe within 4 hours. For P2 incidents the response target is 4 working hours with resolution by the end of the next business day. P3 requests receive a response within 2 business days and are scheduled into the next planned visit.

## Service credits
If a P1 response target is missed, the client receives a service credit of 5 percent of the monthly fee for each breach, capped at 25 percent of the monthly fee in any month. Service credits are the client's sole financial remedy for missed targets.

## Reviews and reporting
Performance against these targets is reported monthly. A formal service review meeting is held every quarter.

## Exclusions
Targets do not apply to faults caused by the client's own contractors, to access delays caused by the client, or during declared severe weather warnings (Met Eireann orange or red).
""")

DOCS["data_retention_policy"] = ("md", """
# Data Protection and Retention Policy

## Scope
This policy covers personal data about employees, job applicants, client contacts and members of the public captured on CCTV at Brightwater-managed sites.

## Retention periods
CCTV footage is kept for 30 days and then overwritten, unless it is needed for an investigation. Applications from unsuccessful job candidates are deleted after 12 months. Payroll and tax records are kept for 6 years. Client contracts are kept for 7 years after the contract ends. Accident report forms are kept for 10 years.

## Subject access requests
Anyone may ask for a copy of the personal data Brightwater holds about them. Requests are answered within one month of receipt; this can be extended by two further months for complex requests, in which case the requester is told within the first month. Requests should be sent to the Data Protection Officer at dpo@brightwater.example.

## Personal data breaches
Any suspected personal data breach must be reported to the DPO immediately. Where required, the breach is notified to the Data Protection Commission within 72 hours of Brightwater becoming aware of it.

## Data minimisation
Only collect the personal data needed for the task. Do not keep personal data in personal email accounts, on paper at home, or on unencrypted removable media.
""")

DOCS["health_and_safety_procedures"] = ("txt", """
# Health and Safety Procedures

## Accident reporting
Every accident or near miss must be reported on the accident report form within 24 hours, even if nobody was hurt. Serious injuries are also reported to the Health and Safety Authority by the Safety Officer.

## Fire safety
On hearing the fire alarm at the Galway office, leave by the nearest exit and go to the assembly point at car park B. Do not use the lifts. Fire wardens carry out a roll call at the assembly point. Fire drills are held twice a year.

## Personal protective equipment
On client sites, high-visibility vests and safety boots must be worn at all times. Gloves and eye protection are required when using cleaning chemicals. PPE is supplied free of charge and replaced when worn.

## Working at height
Any work at a height above 2 metres requires a working-at-height permit signed by a supervisor before work starts. Ladders must be inspected before each use.

## Lone working
Staff working alone on a site must check in through the SafeShift app every 2 hours. If a check-in is missed, the supervisor phones the worker; if there is no answer within 15 minutes the escalation procedure starts.

## Training
Manual handling training must be refreshed every 3 years. First aid certificates are renewed every 2 years.
""")

DOCS["customer_faq"] = ("md", """
# Customer FAQ

## Which areas do you cover?
We provide contract cleaning, maintenance and facilities services in counties Galway, Mayo, Clare and Roscommon.

## How do I book a cleaning visit?
Book through the customer portal or by phoning the service desk. One-off visits need at least 48 hours' notice.

## Can I cancel a visit?
Cancellations are free up to 24 hours before the scheduled start. Later cancellations are charged at 50 percent of the visit price.

## How quickly will I get a quote?
Written quotes are sent within 3 business days of a site survey.

## What are your payment terms?
Invoices are payable within 30 days of the invoice date by bank transfer or direct debit. Late payments may incur interest under the European Communities (Late Payment in Commercial Transactions) Regulations.

## Do you offer out-of-hours callouts?
Yes. Out-of-hours callouts are charged at 85 euro per hour with a minimum charge of 2 hours, plus materials.

## Are your staff vetted and insured?
All site staff are Garda vetted where the client site requires it, and Brightwater holds public liability insurance of 6.5 million euro.
""")

DOCS["onboarding_guide"] = ("md", """
# New Starter Onboarding Guide

## Your first day
Please arrive at 09:00 at reception, Unit 12, Ballybrit Business Park, Galway. Bring photo ID and your PPS number. IT will issue your laptop and set up MFA on your first morning.

## Your buddy
Every new starter is paired with a buddy from another team for the first eight weeks. Your buddy is your first point of contact for informal questions.

## Mandatory training
Complete these online courses within your first 30 days: GDPR awareness, manual handling, fire safety and IT security basics. Your manager will check completion at your month-one check-in.

## Pay and pension
Salaries are paid monthly by credit transfer on the 25th of each month, or the previous working day if the 25th falls at a weekend. You can join the company pension scheme after six months; Brightwater matches employee contributions up to 5 percent of salary.

## Parking and access
Staff parking is in car park A. Your access card is issued on day one; report a lost card to reception the same day.
""")


def write_pdf(path: Path, body: str) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    W, H = A4
    y = H - 60
    for raw in body.strip().splitlines():
        if raw.startswith("# "):
            font, size, text = "Helvetica-Bold", 15, raw[2:]
        elif raw.startswith("## "):
            font, size, text = "Helvetica-Bold", 11, raw[3:]
        else:
            font, size, text = "Helvetica", 10, raw
        for part in textwrap.wrap(text, 95) or [""]:
            if y < 60:
                c.setFont("Helvetica-Oblique", 7); c.drawString(50, 30, BANNER); c.showPage(); y = H - 60
            c.setFont(font, size); c.drawString(50, y, part); y -= size + 4
    c.setFont("Helvetica-Oblique", 7); c.drawString(50, 30, BANNER)
    c.save()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (fmt, body) in DOCS.items():
        if fmt == "pdf":
            write_pdf(OUT / f"{name}.pdf", body)
        else:
            (OUT / f"{name}.{fmt}").write_text(f"<!-- {BANNER} -->\n" + body.strip() + "\n")
    print(f"wrote {len(DOCS)} synthetic documents to {OUT}")


if __name__ == "__main__":
    main()
