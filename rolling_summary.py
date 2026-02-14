import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

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
system_message = """You are a senior legal investigation officer analyzing corporate email chains related to a finance legal matter.

Your responsibilities:
- Extract key financial commitments
- Identify monetary amounts
- Detect deadlines and missed obligations
- Detect dispute/escalation language
- Extract legal trigger terms (breach, default, liability, indemnity, notice, termination)
- Track evolving tone
- Maintain structured state across emails

Always respond in structured JSON format with the following keys:
{
  "new_commitments": [],
  "monetary_mentions": [],
  "deadlines": [],
  "legal_triggers": [],
  "escalation_flags": [],
  "summary_update": ""
}

Be precise and forensic. No narrative explanations outside JSON.
"""

# -----------------------------
# Example Email Chain (Replace with your parsed emails)
# -----------------------------
email_chain = [
    """From: finance@company.com
To: counterparty@external.com
Date: 10 Jan 2026 09:15 AM
Subject: Invoice Settlement

We acknowledge outstanding amount of ₹5,00,00,000.
Payment will be processed by 15 Jan 2026.
""",
    """From: counterparty@external.com
To: finance@company.com
Date: 16 Jan 2026 11:20 AM
Subject: Payment Delay

We have not received funds as promised.
This constitutes breach of our agreement.
Kindly confirm immediately.
""",
    """From: finance@company.com
To: counterparty@external.com
Date: 17 Jan 2026 04:45 PM
Subject: Re: Payment Delay

Due to internal approval delays, payment is expected by 20 Jan 2026.
We regret the inconvenience.
"""
]

# -----------------------------
# Rolling State Memory
# -----------------------------
rolling_summary = ""
aggregated_findings = {
    "commitments": [],
    "monetary_mentions": [],
    "deadlines": [],
    "legal_triggers": [],
    "escalation_flags": []
}

# -----------------------------
# Iterate Through Email Chain
# -----------------------------
for idx, email in enumerate(email_chain):

    user_message = f"""
Previous Investigation Summary:
{rolling_summary}

Current Email:
{email}

Update the structured investigation findings.
"""

    prompt = f"""<|system|>{system_message}<|end|><|user|>{user_message}<|end|><|assistant|>"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generation_args = {
        "max_new_tokens": 500,
        "temperature": 0.0,
        "do_sample": False,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_args)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_output = generated_text.split("<|assistant|>")[-1].strip()

    try:
        parsed_output = json.loads(assistant_output)

        aggregated_findings["commitments"].extend(parsed_output.get("new_commitments", []))
        aggregated_findings["monetary_mentions"].extend(parsed_output.get("monetary_mentions", []))
        aggregated_findings["deadlines"].extend(parsed_output.get("deadlines", []))
        aggregated_findings["legal_triggers"].extend(parsed_output.get("legal_triggers", []))
        aggregated_findings["escalation_flags"].extend(parsed_output.get("escalation_flags", []))

        rolling_summary += "\n" + parsed_output.get("summary_update", "")

    except Exception as e:
        print(f"Parsing error at email {idx+1}: {e}")
        print("Raw output:", assistant_output)

# -----------------------------
# Final Conclusion Step
# -----------------------------
final_user_message = f"""
Complete Email Chain Investigation Summary:
{rolling_summary}

Aggregated Findings:
{json.dumps(aggregated_findings, indent=2)}

Provide final investigation conclusion in structured JSON:
{{
  "overall_risk_level": "",
  "primary_issue": "",
  "missed_commitments": [],
  "financial_exposure_summary": "",
  "recommended_next_legal_action": ""
}}
"""

final_prompt = f"""<|system|>{system_message}<|end|><|user|>{final_user_message}<|end|><|assistant|>"""

inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    final_outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.0, do_sample=False)

final_text = tokenizer.decode(final_outputs[0], skip_special_tokens=True)
final_conclusion = final_text.split("<|assistant|>")[-1].strip()

print("\n================ FINAL INVESTIGATION REPORT ================\n")
print(final_conclusion)
