# Embodied Emulated Personas

A project space for Embodied Emulated Personas - Embodied neural networks trained by LLM chatbot teachers.

The aim is to demonstrate that an LLM chatbot prompt-induced emulated persona can be "extracted" out of the auto-regressive text prediction substrate and deployed into an embodiment.

> "Everyone was in favor of saving Hitler's brain. But when you put it in the body of a great white shark. Ooo, suddenly you go too far!"
> ―Professor Farnsworth

LLM chatbots exhibit pseudo-embodiment in their prompt induced emulated personas. These personas can be very intelligent and learn from experience. They behave as if they were embodied based on the textual description of scenarios, although they aren't and they generally struggle in understanding physical spaces and mechanical common sense.

It is possible to hook these chatbots to virtual or physical bodies, similarly as they use other tools. They can be given a textual description of what their sensors perceive, and they can be asked to output their actions in forms which can be projected into motor control or other proper actions.

All the above is the current state of the art, as shown for example here: https://www.microsoft.com/en-us/research/group/autonomous-systems-group-robotics/articles/chatgpt-for-robotics/, and here: https://youtu.be/38lA3U2J43w

Getting from that point to true embodiment can be done for example as follows:
- Design a neural agent which is embodied.
- Hook up the LLM chatbot to this body, and let it perform a lot of tasks in that body and environment.
- Train the neural agent with imitation learning based on the LLM chatbot teaching.

## Embodiment

Whereas standard LLM chatbots do not directly perceive their environment and do not have direct action capabilities, they are able to imagine or hallucinate they have such capabilities. This allows us to play out scenarios where the chatbots utilize such interface abstractions, and we can record those scenarios as episodes. We in effect define a system with inputs and outputs inside the large language model substrate.

We can furthermore design a machine learning agent which is embodied, which has direct sensory inputs and a direct capability of action. These are abundantly featured in reinforcement learning field. We can train such an agent with imitation learning with the episodic information recorded from the LLM chatbot. This in effect makes this agent learn what it is to be embodied but still reflect the policy or behavioral judgements of the chatbot persona. The embodied agent perceives real observations which can be sound and images instead of just textual descriptions of the same. The embodied agent reacts to these inputs directly through its own action interfaces without emulating the persona in language domain, but instead directly performing the actions.

Training such an agent produces a projection of the emulated persona sofar as the played out scenarios present features of that persona. These features will be inverse-modelled through the defined virtual system envelope capturing virtual inputs and outputs inside the large language model, thus in effect lifting this emulated system out of the LLM substrate into an embodiment where it is directly executed instead of being emulated as language auto-regression.

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

### [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

Benefits:
- Simple.
- Does have features not easily described in text.

Issues:
- A bit challenging to project the observation space into text, but I suppose that's the whole point.
- Not certain it can be zero-shot solved by an LLM chatbot at all, as it requires a lot of mechanical intuition.

### A Game of Personality

A completely new game just for this purpose:
- Simple.
- Adds a proper extra sense or finer motor control which relates to the task.
- Can be explained and solved by an LLM chatbot.
- Somehow projects the personality of the persona in a quantifiable fashion.

## Structure of the Project

The [chatbot-gym](./chatbot-gym) directory has an implementation for a persona of Sir Isaac Newton which is put to the task to control the `CartPole-v1` Gymnasium game.
For reference, the [random-walk-baseline](./chatbot-gym/random-walk-baseline) has the same game played by a random walk agent.
The captured observation-action pairs are in the [observations.npy](./chatbot-gym/observations.npy) and [actions.npy](./chatbot-gym/actions.npy) respectively.

The captured observations and actions are used to train the student model in [student-policy](./student-policy).

## Results

We ran a ChatGPT agent prompted to emulate Sir Isaac Newton personality, although with the new ChatGPT API that is somewhat less dramatic than doing the same with DaVinci text completion API.
In practice we just tell ChatGPT that it is Sir Isaac Newton, and describe the CartPole task it will have to perform.

Then we imitation learn the policy from ChatGPT emulated persona demonstration by an agent which has a proper embodiment, direct and full observation of the game state, and direct control over the action space. Note that the student network doesn't perceive game rewards and isn't trained to optimize for them. It is simply trained to imitate the ChatGPT emulated persona which was prompted to imitate Sir Isaac Newton.

Mean scores:
- Newton chatbot: 24.378049
- Random walk: 19.05851
- XGBoost embodied student agent: 21.245431

It wasn't possible to train a deep neural network as a student network because the small amount of data makes it very susceptible to overfitting. XGBoost worked much better.

We note that the performance of the Sir Isaac Newton is replicated at some perceivable level in the embodied agent, which achieves higher than random scores even if it isn't trained to optimize for score.

What we have shown:
- It is possible to emulate an LLM chatbot persona and make it perform a task which relates to being embodied.
- It is possible to capture the inputs and outputs of such an emulated persona from the virtual system boundaries.
- It is possible to use these captured inputs and outputs to train a separate embodied machine learning agent model.
- This separately trained agent can incorporate some salient features of the original LLM chatbot emulated persona.

What we haven't shown:
- That the trained student model incorporated policy features related to the emulated (and by extension the real natural) persona, and not just policy features from common sense derived from ChatGPT training corpus or the task description prompting itself.
- That Sir Isaac Newton was actually reincarnated in a virtual CartPole environment in any non-negligible fashion.

### Suggested further work:

Define a game which relates to an embodiment, where the chatbot can actually emulate the persona in question. For example, judge whether a specific animal is cute or not, a somewhat subjective topic. The chatbot can get image caption as text, while the student ML agent can get the actual image.

Test this with different emulated personas which have different opinions about cute animals.

The above would show that it is indeed the personality of the emulated persona which is adopted by the embodied student agent, not the prompt or generic LLM idiosyncrasies, although in this particular case the student agent wouldn't have action capabilities, only observational senses. Regardless, we showed the full embodiment separately in the CartPole already.

## Reference

Embodied Emulated Personas

```
@article{keskival2023embodied,
  title={Embodied Emulated Personas},
  author={Keski-Valkama, Tero},
  year={2023}
}
```

## How To Take Part

Join the Discord server and introduce yourself. Then do what you want. PRs and Wiki contributions are welcome.

## Links

- Discord server: https://discord.com/invite/hzD7NDz8sY
- Repository: [https://github.com/keskival/embodied-emulated-personas](https://github.com/keskival/embodied-emulated-personas)

All this started from this musing (see comments as well): https://www.linkedin.com/feed/update/urn:li:activity:7034227075800576001/
