# latency

Latency is a model designed to simulate artificial delays between the user and a large language model (LLM). By introducing controlled latency, conversations can feel more natural and human-like, as instant responses are replaced with realistic pauses similar to those in real human interactions.

##  Dataset

To provide a realistic simulation, the dataset consists of messages with associated latencies and timestamps. Each message is represented as a JSON object with the following fields:

```json
[
    {"latency": 6, "time": 123, "message": "msg1"},
    {"latency": 1, "time": 456, "message": "msg2"},
    {"latency": 3, "time": 789, "message": "msg3"},
]
```

`latency` is the time in minutes that the next person replied to the message
`time` is the current time converted to minutes (0 to 1440)
`message` is the content of the message
