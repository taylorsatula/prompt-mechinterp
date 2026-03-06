Run the prep step of the MI pipeline using the three input files in this directory.

The command is:

```
python -m prompt_mechinterp.prep.inputs \
    --prompt data/inputs/system_prompt.txt \
    --regions data/inputs/regions.json \
    --conversations data/inputs/conversations.json \
    --output data/inputs/test_cases.json
```

After it succeeds, read through the generated `test_cases.json` and verify:
1. The system_regions map has 6 regions (directive, task_entity, task_passage, task_expansion, task_complexity, output_format)
2. The single case has user_regions (conversation_turns, current_message, stored_passages) and response_regions (thinking, entities, passages, expansion, complexity)
3. All char_start/char_end values are non-zero and non-overlapping within their section

Then describe what the next step would be (scp to GPU box and run the engine) based on the README and SKILL.md.
