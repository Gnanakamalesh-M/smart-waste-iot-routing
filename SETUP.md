# SETUP — do this BEFORE the first Claude Code session (Windows cmd)

## 1. Make the folder and copy the kit
```
mkdir smart-waste-iot-routing
cd smart-waste-iot-routing
```
Copy these into it: `CLAUDE.md`, `docs\SPEC.md`, `docs\PHASES.md`.
(Claude Code reads an empty folder otherwise, so SPEC.md must exist first.)

## 2. Git identity BEFORE the first commit
```
git init
git config --local user.name "Gnanakamalesh-M"
git config --local user.email "YOUR_ID+Gnanakamalesh-M@users.noreply.github.com"
git config --show-origin user.name
```
Get your numeric ID + noreply address from GitHub: Settings > Emails.
Tick "Keep my email addresses private" there.

## 3. Turn off Claude co-author lines
Create `.claude\settings.json` in the repo:
```
{
  "includeCoAuthoredBy": false
}
```
Newer Claude Code versions may use an `attribution` setting instead. If commits still show a
Co-Authored-By line, check the Claude Code docs for your version. CLAUDE.md also forbids it.

## 4. Python 3.11 venv
```
py -3.11 --version
py -3.11 -m venv .venv
.venv\Scripts\activate
```
If 3.11 is not installed, install it first. CI uses 3.11, so test on 3.11.

## 5. GitHub repo
Create at github.com/new (NOT /import). Name: smart-waste-iot-routing.
Leave README, .gitignore, and license UNTICKED. Then:
```
git remote add origin https://github.com/Gnanakamalesh-M/smart-waste-iot-routing.git
```
If push gives 403, remove old github entries in Windows Credential Manager and log in again.
Push yourself after reviewing (Claude Code is told never to push).

## 6. Start
```
claude
```
Press Shift+Tab to enter plan mode, then paste the Phase 0 prompt from `docs\PHASES.md`.

## Quick checks
- `findstr` instead of `grep`.
- Always `.venv\Scripts\activate` before pytest.
- GitHub "Internal Server Error" on push: check githubstatus.com, wait, retry once.
