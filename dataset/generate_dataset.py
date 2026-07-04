"""
generate_dataset.py
--------------------
Generates a realistic, synthetic email dataset for the Email Spam Detection
project. Produces a balanced set of Spam and Ham (legitimate) emails with
metadata features, and writes it to dataset/spam_dataset.csv.

This script is provided for transparency/reproducibility. The dataset has
already been generated and saved as spam_dataset.csv, so you do not need to
re-run this unless you want to regenerate it (results will differ slightly
due to randomness, unless SEED is kept constant).
"""

import random
import re
import pandas as pd
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_RECORDS = 5000  # total emails (balanced spam/ham)

# ---------------------------------------------------------------------------
# 1. Building blocks for SPAM emails
# ---------------------------------------------------------------------------
spam_openers = [
    "CONGRATULATIONS! You have been selected",
    "URGENT: Your account requires immediate action",
    "Winner Notification: You have WON",
    "Limited Time Offer just for YOU",
    "Act now before it's too late",
    "FINAL NOTICE regarding your payment",
    "You are our lucky visitor today",
    "Claim your FREE reward now",
    "Exclusive deal available for 24 hours only",
    "Your package could not be delivered",
]

spam_bodies = [
    "a brand new iPhone 15 Pro absolutely free of cost. Click the link below to claim your prize before it expires!",
    "a cash prize of $1,000,000 in our international lottery. Send your bank details to process the transfer immediately.",
    "an exclusive investment opportunity with guaranteed 500% returns in just 7 days. Limited slots available, act fast!",
    "a free trial of our miracle weight loss pills. Lose 10kg in one week, no diet or exercise required!",
    "a special discount of 90% off on luxury watches. Hurry, offer valid only for the next few hours!",
    "verification that your bank account has been suspended. Click here to verify your details and avoid permanent closure.",
    "a job offer paying $5000 per week working from home, no experience needed. Reply with your personal details to start.",
    "access to premium software completely free. Download the attached file now to activate your license.",
    "a refund of your recent purchase. Please confirm your credit card number and CVV to process the refund.",
    "an inheritance claim from a distant relative worth $2,500,000. Contact our lawyer immediately with your details.",
    "a chance to win a brand new car by simply clicking the link and filling out a short form with your details.",
    "cheap replica designer handbags shipped worldwide, buy 1 get 3 free, limited stock available today only.",
]

spam_ctas = [
    "Click here now: http://claim-your-prize-now.win/reward",
    "Visit http://free-money-fast.biz/claim to proceed immediately.",
    "Follow this link http://secure-bank-verify.info/login to verify.",
    "Download now: http://get-rich-quick.top/offer",
    "Reply immediately to http://job-offer-cash.click/apply with your bank details.",
    "Confirm here: http://account-alert-update.site/confirm",
    "Order now at http://luxury-deals-cheap.shop/buy before stock runs out!!!",
    "Call our toll-free hotline NOW or click http://winner-notification.club/claim",
]

spam_signoffs = [
    "This offer will not be repeated. Act NOW!!!",
    "Don't miss this once in a lifetime opportunity!",
    "Time is running out, claim your reward TODAY.",
    "100% guaranteed, no risk involved!",
    "Hurry! Only a few slots remaining!!!",
]

spam_senders = [
    "prizeteam@lottery-winners.win", "support@bank-security-alert.info",
    "offers@luxury-deals-cheap.shop", "no-reply@free-money-fast.biz",
    "hr@work-from-home-jobs.top", "admin@account-verification.site",
    "deals@discount-mega-sale.club", "winner@international-lottery.org",
]

spam_subjects = [
    "You Have WON $1,000,000!!!", "URGENT: Verify Your Account Now",
    "FREE iPhone 15 Pro - Claim Now", "Congratulations Lucky Winner",
    "Limited Time Offer - 90% OFF", "Your Refund Is Waiting",
    "Work From Home - Earn $5000/Week", "FINAL NOTICE: Account Suspension",
    "Exclusive Investment Opportunity", "Claim Your FREE Reward Today",
]

# ---------------------------------------------------------------------------
# 2. Building blocks for HAM (legitimate) emails
# ---------------------------------------------------------------------------
ham_openers = [
    "Hi Team,", "Hello,", "Dear Sir/Madam,", "Hi John,", "Good morning,",
    "Hey,", "Dear Customer,", "Hi Professor,", "Hello Everyone,", "Dear Manager,",
]

ham_bodies = [
    "I hope this email finds you well. I wanted to follow up on the meeting we had last week regarding the project timeline.",
    "Please find attached the quarterly report for your review. Let me know if you have any questions or need further clarification.",
    "Thank you for your prompt response. I have updated the document as per your suggestions and shared it in the drive folder.",
    "This is a reminder that our team meeting is scheduled for tomorrow at 10 AM in the conference room. Please come prepared.",
    "I am writing to confirm our appointment scheduled for next Monday at 3 PM. Please let me know if this time still works for you.",
    "Attached is the invoice for the services rendered last month. Kindly process the payment at your earliest convenience.",
    "Just a quick note to say thank you for your help with the project. Your input was extremely valuable to the team.",
    "Could you please share the updated syllabus for the upcoming semester? I would like to prepare my study plan accordingly.",
    "Your order has been shipped and is expected to arrive within 3-5 business days. Track your shipment using the order ID provided.",
    "I would like to schedule a call to discuss the recent changes in the project scope. Please let me know your availability this week.",
    "The system maintenance is scheduled for this weekend. Services may be temporarily unavailable between 1 AM and 4 AM.",
    "Please review the attached minutes of the meeting and let me know if any corrections are needed before we circulate them.",
]

ham_signoffs = [
    "Best regards,", "Thank you,", "Sincerely,", "Warm regards,", "Kind regards,",
    "Regards,", "Looking forward to your reply,", "Thanks and regards,",
]

ham_names = [
    "Priya Sharma", "Rahul Verma", "Ananya Iyer", "Karthik Raj", "Sneha Nair",
    "Arjun Menon", "Divya Krishnan", "Vikram Singh", "Meera Pillai", "Suresh Kumar",
]

ham_senders = [
    "priya.sharma@company.com", "rahul.verma@university.edu",
    "hr@techsolutions.com", "support@onlinestore.com",
    "professor.rao@college.edu", "accounts@finance-services.com",
    "manager@projectteam.org", "admin@office-portal.com",
    "noreply@shipping-service.com", "team@workplace.net",
]

ham_subjects = [
    "Meeting Follow-up", "Quarterly Report Attached", "Appointment Confirmation",
    "Project Timeline Update", "Invoice for Last Month", "Weekly Team Meeting Reminder",
    "Syllabus for Upcoming Semester", "Your Order Has Been Shipped",
    "Scheduled System Maintenance", "Minutes of the Meeting",
]


def count_links(text):
    return len(re.findall(r"http[s]?://", text))


def count_special_chars(text):
    return len(re.findall(r"[!$%*#@]", text))


def count_capitals(text):
    return sum(1 for c in text if c.isupper())


ham_promo_snippets = [
    "As a valued customer, enjoy 20% off your next purchase with code SAVE20.",
    "Don't miss our end of season sale, discounts available this week only.",
    "Your subscription renewal is due soon, click here to manage your account.",
    "We have a special offer just for our newsletter subscribers this month.",
    "Limited seats left for our upcoming webinar, register using the link below.",
]

spam_subtle_bodies = [
    "we noticed unusual activity on your account and recommend confirming your details as a precaution.",
    "our records show your subscription is about to expire, please update your payment information soon.",
    "a business proposal that may interest you, kindly review the attached document and share your thoughts.",
    "your recent order requires additional verification before it can be processed further.",
]


def _maybe_add_noise(text: str) -> str:
    if random.random() < 0.15 and " the " in text:
        text = text.replace(" the ", "  the ", 1)
    if random.random() < 0.10:
        words = text.split()
        if len(words) > 6:
            cut = random.randint(len(words) - 3, len(words))
            text = " ".join(words[:cut])
    return text


def make_spam_email():
    opener = random.choice(spam_openers)
    if random.random() < 0.25:
        body = random.choice(spam_subtle_bodies)
        cta = random.choice(spam_ctas) if random.random() < 0.6 else ""
    else:
        body = random.choice(spam_bodies)
        cta = random.choice(spam_ctas)
    signoff = random.choice(spam_signoffs) if random.random() < 0.8 else ""
    text = f"{opener} {body} {cta} {signoff}".strip()
    text = _maybe_add_noise(text)
    subject = random.choice(spam_subjects)
    sender = random.choice(spam_senders)
    return text, subject, sender


def make_ham_email():
    opener = random.choice(ham_openers)
    if random.random() < 0.20:
        body = random.choice(ham_promo_snippets)
    else:
        body = random.choice(ham_bodies)
    signoff = random.choice(ham_signoffs)
    name = random.choice(ham_names)
    text = f"{opener} {body} {signoff} {name}"
    text = _maybe_add_noise(text)
    subject = random.choice(ham_subjects)
    sender = random.choice(ham_senders)
    return text, subject, sender


def build_dataset(n_records=N_RECORDS):
    records = []
    n_each = n_records // 2

    for i in range(n_each):
        text, subject, sender = make_spam_email()
        records.append({
            "Email_ID": f"EML{i+1:05d}",
            "Email_Text": text,
            "Sender": sender,
            "Subject": subject,
            "Message_Length": len(text),
            "Number_of_Links": count_links(text),
            "Number_of_Special_Characters": count_special_chars(text),
            "Number_of_Capital_Letters": count_capitals(text),
            "Label": "Spam",
        })

    for i in range(n_each):
        text, subject, sender = make_ham_email()
        records.append({
            "Email_ID": f"EML{n_each + i + 1:05d}",
            "Email_Text": text,
            "Sender": sender,
            "Subject": subject,
            "Message_Length": len(text),
            "Number_of_Links": count_links(text),
            "Number_of_Special_Characters": count_special_chars(text),
            "Number_of_Capital_Letters": count_capitals(text),
            "Label": "Ham",
        })

    df = pd.DataFrame(records)

    # Simulate realistic annotation noise: in real-world labelled datasets,
    # a small percentage of labels are ambiguous or mislabeled (e.g. a
    # legitimate promotional email marked spam by a human annotator, or a
    # borderline spam message that slipped through as ham). Flipping ~3%
    # of labels avoids an artificially perfect, trivially-separable dataset
    # and produces more realistic (<100%) model accuracy.
    noise_frac = 0.03
    n_noisy = int(len(df) * noise_frac)
    noisy_idx = df.sample(n=n_noisy, random_state=SEED).index
    df.loc[noisy_idx, "Label"] = df.loc[noisy_idx, "Label"].map({"Spam": "Ham", "Ham": "Spam"})

    # Shuffle rows so spam/ham are not in blocks
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    # Re-issue Email_IDs post-shuffle to keep them sequential and unique
    df["Email_ID"] = [f"EML{i+1:05d}" for i in range(len(df))]
    return df


if __name__ == "__main__":
    df = build_dataset()
    out_path = "spam_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Dataset generated: {len(df)} records -> {out_path}")
    print(df["Label"].value_counts())
