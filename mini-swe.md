# Mini SWE

Details on how to use model with local inference and mini swe agent harness.

## Install harness

```
uv tool install mini-swe-agent
```

## Mini SWE Agent config

Save this config to `~/miniswe.yaml`

```yaml
agent:
  system_template: |
    You are a tiny shell agent. Reply with one brief, simple THOUGHT, then exactly
    one command in this format:
    ```mswea_bash_command
    command
    ```
    The command must solve the task. Do not add anything else.
  instance_template: |
    Write the Bash command for this request:

    {{task}}
  step_limit: 1
  cost_limit: 0
  mode: confirm

model:
  model_class: litellm_textbased
  model_name: openai/mikeoxmaul/zmeeust-miniswe
  cost_tracking: ignore_errors
  model_kwargs:
    api_base: http://localhost:8000/v1
    api_key: dummy
    temperature: 0.1
    max_tokens: 128
```

## Inference

start local inference:

```
uv run --with "transformers[serving],requests" transformers serve \
  mikeoxmaul/zmeeust-miniswe \
  --reasoning auto \
  --device auto \
  --port 8000
```

## Run Agent

```
mini \
  -c ~/miniswe.yaml \
  -t 'list all Python files recursively'
```

Or in interactive mode:

```
mini -c ~/miniswe.yaml
```
