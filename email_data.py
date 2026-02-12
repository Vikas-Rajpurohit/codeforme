import re
import html

def extract_email_headers(text):
    # Fix encoded HTML characters
    text = html.unescape(text)

    # Pattern to capture full header block
    pattern = r"From:.*?Subject:.*?(?=\r?\n\r?\n|From:|$)"

    blocks = re.findall(pattern, text, flags=re.DOTALL)

    extracted_data = []

    for block in blocks:
        data = {}

        from_match = re.search(r"From:\s*(.*)", block)
        sent_match = re.search(r"Sent:\s*(.*)", block)
        to_match = re.search(r"To:\s*(.*)", block)
        cc_match = re.search(r"Cc:\s*(.*)", block)
        subject_match = re.search(r"Subject:\s*(.*)", block)

        data["from"] = from_match.group(1).strip() if from_match else None
        data["sent"] = sent_match.group(1).strip() if sent_match else None
        data["to"] = to_match.group(1).strip().split(";") if to_match else []
        data["cc"] = cc_match.group(1).strip().split(";") if cc_match else []
        data["subject"] = subject_match.group(1).strip() if subject_match else None

        extracted_data.append(data)

    return extracted_data
