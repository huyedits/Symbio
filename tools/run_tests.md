---
name: run_tests
family: code
group: code
seeded: 00351b20116eb01e
---

Run this project's own pytest suite and get a pass/fail report. This is how you verify any change you make to symbio's source: edit with edit_file, call run_tests, and let the output decide whether the fix worked. Accepts targets 'tests' (the main suite, default) and 'bench' (the benchmark-harness tests); with several, they run in one pytest invocation. A run takes minutes — call it after an edit, not on a guess.

```json
{
  "type": "object",
  "properties": {
    "targets": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Which suite(s): 'tests', 'bench'. Default: all of tests/."
    }
  }
}
```
