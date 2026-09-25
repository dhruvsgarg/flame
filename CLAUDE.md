# Repo instructions

## Every session: read the readiness docs first

`lib/python/examples/_metadata/ROBUST_FL_READINESS.md` (rules, shared lessons/tripwires, shared queue), then
the active child: `FELIX_READINESS.md` (current focus) or `FLUXTUNE_READINESS.md`. Resume from the child's
"Next steps". Standing instructions are recorded there, not in agent memory.

## Before any commit + push

Whenever the user says to commit and push (or just "commit"), first:

1. Review the diff and rewrite any verbose/rambling comments added this session into crisp
   one-liners — same logical content, minimum words. Don't touch comments you didn't just add.
2. Keep the diff minimal: don't re-wrap or reformat lines you aren't otherwise changing.
3. Commit title: under 8 words. Commit message body: under 50 words.

Then proceed with the normal commit flow (stage, commit, confirm before push).
