# Embodied Emulated Personas

A project space for Embodied Emulated Personas - Embodied neural networks trained by LLM chatbot teachers.

The aim is to demonstrate that an LLM chatbot prompt-induced emulated persona can be "extracted" out of the auto-regressive text prediction substrate and deployed into an embodiment.

> "Everyone was in favor of saving Hitler's brain. But when you put it in the body of a great white shark. Ooo, suddenly you go too far!"
> ―Professor Farnsworth

LLM chatbots exhibit pseudo-embodiment in their prompt induced emulated personas. These personas can be very intelligent and learn from experience. They behave as if they were embodied based on the textual description of scenarios, although they aren't and they generally struggle in understanding physical spaces and mechanical common sense.

It is possible to hook these chatbots to virtual or physical bodies, similarly as they use other tools. They can be given a textual description of what their sensors perceive, and they can be asked to output their actions in forms which can be projected into motor control or other proper actions.

All the above is the current state of the art, as shown for example here: https://www.microsoft.com/en-us/research/group/autonomous-systems-group-robotics/articles/chatgpt-for-robotics/

Getting from that point to true embodiment can be done for example as follows:
- Design a neural agent which is embodied.
- Hook up the LLM chatbot to this body, and let it perform a lot of tasks in that body and environment.
- Train the neural agent with imitation learning based on the LLM chatbot teaching.

## Plan

The goal is to get a proof-of-concept done, nothing too fancy. For this, we can leverage simple Farama Gymnasium environments: https://gymnasium.farama.org/

The proof-of-concept will demonstrate:
- A neural agent learning common sense based embodied tasks from chatbot demonstrations.
- It demonstrates that an LLM chatbot prompt-induced emulated persona can be "extracted" out of the auto-regressive text prediction substrate and deployed into an embodiment.

Requirements for the embodiment:
- Needs to have at least one proper sense which can be projected to a textual description.
- Needs to have an action space which can be controlled by text.
- Needs to be simple enough to be controlled by a relatively small neural network.

Requirements for the environment:
- Needs to generate common sense tasks in large numbers.
- The tasks need to be describeable to chatbots which can utilize their common sense and judgement to perform well.

Possible environments:

### [Blackjack](https://gymnasium.farama.org/environments/toy_text/blackjack/)

Benefits:
- Simple, won't require a huge neural network or long training.
- Can be described simply as text.
- Shows characteristics of the persona in how much risk they tolerate. This can be varied with prompting.

Issues:
- Doesn't really add a proper extra sense, as the observations can be fully described in text.

### [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

Benefits:
- Simple.
- Does have features not easily described in text.

Issues:
- A bit challenging to project the observation space into text, but I suppose that's the whole point.

## How To Take Part

Join the Discord server and introduce yourself. Then do what you want. PRs and Wiki contributions are welcome.

## Links

- Discord server: https://discord.com/invite/hzD7NDz8sY

All this started from this musing (see comments as well): https://www.linkedin.com/feed/update/urn:li:activity:7034227075800576001/
