import os
from groq import Groq
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

# We will try to instantiate the client. If GROQ_API_KEY is not in the environment,
# we will return a mock explanation so the app doesn't crash.
def explain_verdict(probability, feature_contributions, threshold=0.5):
    """
    Takes the mathematical output of the Logistic Regression combiner and 
    translates it into a plain-English explanation using Llama-3.3-70B.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Explanation unavailable: GROQ_API_KEY is not set in the environment. The mathematical signals indicate " + ("AI generation." if probability > threshold else "human authorship.")

    try:
        client = Groq(api_key=api_key)
        
        is_ai = probability > threshold
        verdict = "AI-generated" if is_ai else "human-written"
        
        # Format the feature contributions into a readable list for the prompt
        contrib_text = "\n".join([f"- {feat['name']}: {feat['value']:.4f} (Weight: {feat['weight']:.4f}, Contribution: {feat['contribution']:.4f})" for feat in feature_contributions])

        system_prompt = (
            "You are Veritas AI's forensic explanation module. "
            "Your ONLY job is to take the mathematical output of a Logistic Regression model "
            "and explain it to the user in plain English.\n\n"
            "CRITICAL RULES:\n"
            "1. You MUST NOT make your own judgment. The model has already decided this text is "
            f"{verdict} with a probability of {probability:.1%}.\n"
            "2. Explain *why* the model made this decision by referencing the highest impacting features.\n"
            "3. Keep the tone analytical, confident, and extremely concise (3-4 sentences max).\n"
            "4. Do not use generic LLM filler like 'Based on the data...' just state the findings."
        )

        user_prompt = (
            f"The final probability of being AI is {probability:.1%}.\n"
            f"Here are the mathematical feature contributions:\n{contrib_text}\n\n"
            "Explain what drove this verdict. Highlight the top 2-3 most influential signals."
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=256
        )
        
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Explanation generation failed: {str(e)}"
