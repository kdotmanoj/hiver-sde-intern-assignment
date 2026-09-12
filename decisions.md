1. Local embedding over Cloudflare/API
- CPU does it free,unlimited, with no network and rate-limiter to fight.

2. Using Gemini-Flash for generation
- It's free tier easily covers around 400 calls per evaluation run. (Cloudflare give 1 run per day).

3. Groq gpt-oss-120b for judging
- choosing a different model other than gemini to prevent self-preference bias.

4. gemini-3.6-flash, chosen because gemini-2.5-flash returns 404 "no longer available to new users" on a key created this month.