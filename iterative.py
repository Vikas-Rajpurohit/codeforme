import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.random.manual_seed(0)

model_path = "microsoft/Phi-4-mini-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# -----------------------------
# System Prompt (Legal Officer)
# -----------------------------
system_message = """You are a senior legal investigation officer reviewing a finance-related email chain.

Your task:
- Maintain a running summary of the entire conversation.
- Each time, refine and enrich the previous summary using the new email.
- Preserve key financial discussions, commitments, disputes, legal terms, deadlines, and tone shifts.
- Keep the summary concise but comprehensive.
- Do NOT repeat raw email content.
- Produce a single coherent updated summary each time.
"""

# -----------------------------
# Example Email Chain (Replace with your actual parsed emails)
# -----------------------------
email_chain = [
    """From: finance@company.com
Date: 10 Jan 2026
Subject: Invoice Settlement

We acknowledge outstanding amount of ₹5,00,00,000.
Payment will be processed by 15 Jan 2026.
""",
    """From: counterparty@external.com
Date: 16 Jan 2026
Subject: Payment Delay

We have not received funds as promised.
This constitutes breach of our agreement.
Kindly confirm immediately.
""",
    """From: finance@company.com
Date: 17 Jan 2026
Subject: Re: Payment Delay

Due to internal approval delays, payment is expected by 20 Jan 2026.
We regret the inconvenience.
"""
]

# -----------------------------
# Rolling Summary Initialization
# -----------------------------
running_summary = "No prior summary."

# -----------------------------
# Iterative Enrichment
# -----------------------------
for idx, email in enumerate(email_chain):

    user_message = f"""
Previous Summary:
{running_summary}

New Email:
{email}

Update and enrich the overall conversation summary considering the new email.
Return only the updated full summary.
"""

    prompt = f"""<|system|>{system_message}<|end|><|user|>{user_message}<|end|><|assistant|>"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generation_args = {
        "max_new_tokens": 400,
        "temperature": 0.0,
        "do_sample": False,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_args)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    updated_summary = generated_text.split("<|assistant|>")[-1].strip()

    running_summary = updated_summary

    print(f"\n------ Updated Summary After Email {idx+1} ------\n")
    print(running_summary)

# -----------------------------
# Final Summary Output
# -----------------------------
print("\n================ FINAL CONSOLIDATED SUMMARY ================\n")
print(running_summary)
