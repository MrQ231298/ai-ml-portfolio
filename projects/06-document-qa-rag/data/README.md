# Data folder — SYNTHETIC client document pack

**Everything in `client_docs/` is SYNTHETIC.** Brightwater Facilities Ltd
is an invented Galway facilities-management company. Its policies,
figures, e-mail addresses (`@brightwater.example`), product names
(SpendDesk, SafeShift) and the address in Ballybrit Business Park are
fictional and were written for this demo by `src/generate_client_docs.py`.
Every file carries a "SYNTHETIC EXAMPLE DOCUMENT" banner.

| File | Format | Content |
| --- | --- | --- |
| `employee_handbook.md` | Markdown | hours, leave, sick leave, probation, remote work, notice |
| `expenses_policy.pdf` | PDF | claims, approval limits, mileage, subsistence, hotels, cards |
| `it_security_policy.md` | Markdown | passwords/MFA, devices, phishing, patching, backups |
| `service_level_agreement.pdf` | PDF | service hours, priorities, response targets, service credits |
| `data_retention_policy.md` | Markdown | retention periods, subject access requests, breaches |
| `health_and_safety_procedures.txt` | Text | accidents, fire, PPE, working at height, lone working |
| `customer_faq.md` | Markdown | coverage, bookings, cancellations, payment, callouts |
| `onboarding_guide.md` | Markdown | first day, buddy, training, pay and pension, parking |

Why synthetic: a real client's internal documents are confidential and
often contain personal data. The point of the demo is that the same code
runs unchanged on a client's own folder (the app accepts uploads), on
the client's own machine.

Some figures deliberately echo Irish practice (for example the 72-hour
breach notification to the Data Protection Commission and the one-month
subject-access deadline under GDPR), but the documents are not legal or
HR advice.
