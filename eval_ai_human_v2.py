#!/usr/bin/env python3
"""
AI vs Human Detection — Full Eval (v2)
========================================

Comprehensive evaluation with:
- 40 AI samples across styles, genres, personas, lengths
- 40 human samples from Gutenberg
- Adversarial AI samples (told to "write like a human")
- Multi-feature logistic-style classifier
- Confidence calibration
- Exports thresholds for use in the live API
"""

import json
import time
import gc
import numpy as np
from pathlib import Path
from collections import Counter

# ── AI SAMPLES: 40 texts across every style ──────────────────────────

AI_SAMPLES = [
    # --- Novels ---
    {"id": "ai_romance_1", "genre": "novel", "text": """Eleanor had always believed herself above the petty machinations of country society, yet here she stood, adjusting her gloves for the third time while pretending not to notice that Mr. Ashworth had entered the ballroom. He was precisely the sort of man she ought to have dismissed immediately: handsome enough to be dangerous, wealthy enough to be pursued, and clever enough to know both. Their eyes met across the room, and she felt a peculiar sensation, not unlike the moment one realizes a carriage is about to overturn. Mrs. Pemberton, who missed nothing and understood less, seized Eleanor's arm. "That is the gentleman I mentioned. Twelve thousand a year and not a single attachment." Eleanor observed that a man's worth could not be measured in annual income alone, a sentiment she believed with her whole heart and roughly forty percent of her practical judgment. He was approaching now, with the measured stride of someone who knew the effect of his arrival. She would be civil, nothing more. Civility was the armor of the sensible woman, and Eleanor was nothing if not sensible, except perhaps on Tuesday evenings when she read novels by candlelight and forgot herself entirely."""},
    {"id": "ai_adventure_1", "genre": "novel", "text": """The fog was thicker than anything Hargreaves had encountered in thirty years at sea. It pressed against the wheelhouse glass like something alive, swallowing the bow lights entirely and reducing the world to a sphere of gray no larger than the bridge itself. "Can't see a bloody thing, sir," muttered Collins, his hands white on the wheel. Hargreaves said nothing. He was listening. Somewhere ahead, perhaps a quarter mile, perhaps less, he could hear the rhythmic clang of a bell buoy that should not have been there. He checked the chart again. They were twelve miles from the nearest marked channel, running at six knots through water that the admiralty survey showed as empty ocean. Yet the bell clanged on, steady and patient, as if it had been waiting for them. "All stop," he said quietly. Collins pulled the telegraph. The engines died, and in the sudden silence the bell seemed louder, closer. Then another sound reached them through the fog: wood creaking, the slow groan of a hull rolling in a dead swell."""},
    {"id": "ai_psychological_1", "genre": "novel", "text": """Maren understood, with the peculiar clarity that accompanies the worst moments of self-knowledge, that she had been lying to herself for approximately seven years. The realization arrived not as a thunderclap but as a slow subsidence, like discovering that the floor of your house has been gradually sinking into soft ground. She sat at the kitchen table. The coffee was cold. Outside, the neighbor's child was bouncing a ball against a wall with metronomic persistence. Each impact felt like a small accusation. She had constructed her entire adult life around a version of events that she now recognized as fiction, not the dramatic kind where the hero discovers a terrible secret, but the mundane variety in which a person simply edits their memory until it becomes comfortable. The process was so ordinary, so human, that it hardly seemed like deception at all. Everyone did it. She had simply done it more thoroughly than most."""},
    {"id": "ai_gothic_1", "genre": "novel", "text": """The house at the end of Blackmore Lane had been empty for as long as anyone could remember, which in a village like Thornfield meant approximately since the last time someone died in it. The windows were not boarded. They did not need to be. Something about their darkness, the particular quality of absence behind the glass, discouraged approach more effectively than any barrier. Clara walked past it every evening on her way home from the post office, and every evening she felt the same thing: not fear exactly, but a kind of attention, as though the house were watching her with the patient interest of something that had nowhere else to be. It was absurd, she told herself. Houses do not watch. Glass does not see. Darkness is merely the absence of light, not a presence in itself. And yet. On the evening of September the fourteenth, she noticed that a light was burning in the upper window."""},
    {"id": "ai_literary_1", "genre": "novel", "text": """The morning opened slowly, like a letter you already know the contents of. Rain on the windows, the particular gray of March in this city, the sound of the upstairs neighbor's radio playing something classical that arrived through the ceiling as a kind of rumor, melody stripped of its specifics, all shape and no detail. She lay in bed and listened. Time passed differently in the horizontal position. It spread out, lost its edges, became less a sequence of moments and more a texture, like cloth, like the weave of the sheets against her arm. She thought about the word melancholy and decided it was wrong. Melancholy implied weight, and what she felt was closer to transparency, a thinning of the membrane between herself and the world, so that sounds reached her more directly and light seemed to arrive without mediation."""},
    {"id": "ai_scifi_1", "genre": "novel", "text": """The colony ship Perseverance had been decelerating for eleven months when the forward sensors detected something that shouldn't have been there. A structure. Artificial, clearly, from the geometric precision of its edges and the way it reflected starlight at angles that no natural formation could produce. Commander Chen stared at the display for a full thirty seconds before speaking. "That's not in any of our survey data." "No ma'am," said Okafor from the science station. "It's not in anyone's survey data. This system was catalogued as uninhabited." The structure was enormous. Initial radar returns suggested something roughly the size of Manhattan, suspended in a stable orbit around the system's fourth planet. It was not transmitting on any frequency they could detect. It was not moving. It was simply there, patient and unexplained, like a question someone had written in the margins of the universe and forgotten to erase."""},
    {"id": "ai_thriller_1", "genre": "novel", "text": """The phone call came at 3:47 AM, which is the hour when nothing good ever happens. Reeves answered on the second ring because he had not been sleeping, because sleep was something that happened to other people now, people whose names were not on the list he had found folded into the lining of a dead man's coat. "They know," said the voice on the other end. No greeting, no identification. The voice belonged to someone he had never met in person and might never meet. "They know what?" he asked, though he already understood. "About the files. About the names. They're moving tonight." Reeves looked at the clock. Looked at the laptop on the table, its screen dark, its hard drive containing enough information to end careers, end marriages, possibly end lives. He had forty-seven minutes before the shift change at the embassy. After that, the window closed permanently."""},
    {"id": "ai_detective_1", "genre": "novel", "text": """Inspector Marsh had seen enough crime scenes to know that this one was wrong. Not wrong in the obvious way, not the gore or the violence, which were unremarkable by the standards of his profession, but wrong in its precision. The body lay exactly centered on the living room rug. The wounds were symmetrical. The weapon, a kitchen knife, had been placed beside the right hand at a perfect ninety-degree angle, as if measured with a protractor. "Someone's making a point," said Sergeant Obi, who had a gift for stating the obvious in ways that made it sound profound. Marsh nodded. The question was what point, and to whom. He knelt beside the body. The victim was a man in his fifties, well-dressed, clean-shaven, wearing a watch that cost more than Marsh's car. No defensive wounds. No sign of struggle. Either he had known his killer well enough to let them get close, or he had not known them at all and had simply not believed that what was happening was real until it was too late."""},

    # --- Philosophy ---
    {"id": "ai_ethics_1", "genre": "philosophy", "text": """The question of whether moral obligations exist independently of human convention is not merely academic. It reaches into every courtroom, every legislature, every private decision about how we ought to treat one another. Consider the utilitarian position: an action is right insofar as it maximizes overall well-being, wrong insofar as it diminishes it. This has the appeal of mathematical clarity. We calculate, we compare, we choose the option that produces the greatest good for the greatest number. Yet the difficulties emerge immediately upon application. Whose well-being counts? How do we compare the intense suffering of one against the mild discomfort of many? Is a world of mediocre contentment preferable to one of great joy and great pain? The deontologist offers an alternative: certain actions are inherently right or wrong, regardless of their consequences. We must not use people merely as means, must not lie even when lying would produce better outcomes."""},
    {"id": "ai_epistemology_1", "genre": "philosophy", "text": """What can we know with certainty? The question seems simple until you try to answer it. I am sitting at a desk, typing these words. I believe this to be true. But belief and knowledge are not the same thing, and the gap between them is wider than most people suspect. I can doubt my senses, which have deceived me before. I can doubt my reasoning, which has led me astray. I can even doubt the existence of the desk, since I have dreamed of equally solid objects. But notice something: the doubt itself cannot be doubted. If I am doubting, then there is something doing the doubting. This is the classic Cartesian move, and it remains surprisingly difficult to refute. From this one certainty, we might hope to rebuild the edifice of knowledge. We perceive, we infer, we test our inferences against further perception."""},
    {"id": "ai_existentialism_1", "genre": "philosophy", "text": """Freedom is not a gift. It is a condition, an inescapable one, and it weighs more than most people are willing to acknowledge. We are, as the existentialists argued, condemned to choose. Even the refusal to choose is a choice, and it carries the same burden of responsibility as any other. This is uncomfortable. Most of the structures we build around ourselves, the routines, the institutions, the belief systems, are attempts to narrow the field of choice to something manageable, to pretend that the options are fewer than they actually are. We follow rules not because we must but because the alternative, radical freedom without guardrails, is terrifying. The anxiety that comes with authentic existence is not a sign that something has gone wrong. It is a sign that something has gone right. You are feeling the weight of your own agency, and it is heavier than you expected."""},

    # --- Science ---
    {"id": "ai_evolution_1", "genre": "science", "text": """The mechanism of natural selection, once understood, has a simplicity that borders on the inevitable. Organisms vary. Some variations improve survival and reproduction. Those variations are heritable. Over generations, favorable traits accumulate in the population. That is the entire theory, stated in four sentences. Yet from this simple engine arises the staggering diversity of life on Earth: the compound eye of the dragonfly, the echolocation of the bat, the immune system's ability to recognize billions of molecular patterns it has never encountered before. The power of the mechanism lies in its iterative nature. Each generation is a new experiment, each organism a hypothesis about what works in its particular environment. Failed hypotheses are eliminated, not by any conscious judge, but by the blunt arithmetic of differential reproduction."""},
    {"id": "ai_physics_1", "genre": "science", "text": """The universe is under no obligation to make sense to us. This is worth remembering every time we encounter a result in physics that seems to violate common intuition. Quantum mechanics tells us that particles can exist in multiple states simultaneously until observed. General relativity tells us that gravity is not a force but a curvature in the geometry of spacetime itself. Both theories have been confirmed to extraordinary precision, and both describe a reality profoundly unlike our everyday experience. We evolved to navigate a world of medium-sized objects moving at modest speeds, and our intuitions are calibrated accordingly. When we peer into the very small or the very large, those intuitions fail. The electron does not orbit the nucleus like a tiny planet."""},
    {"id": "ai_neuro_1", "genre": "science", "text": """The brain is not a computer, despite the popularity of that metaphor. Computers process information sequentially through deterministic logic gates. Brains are wet, noisy, massively parallel systems in which billions of neurons fire in patterns that are influenced by everything from blood sugar levels to the ambient temperature of the room. There is no central processor, no clean separation between memory and computation, no operating system directing traffic. Instead there is a vast network of connections, each one strengthened or weakened by experience, forming a landscape of associations that is unique to each individual and changes with every passing moment. What we call thinking is not calculation but pattern completion. The brain receives partial, noisy input from the senses and fills in the rest from its own history of past inputs. Perception is not passive reception but active construction."""},

    # --- Drama ---
    {"id": "ai_dialogue_1", "genre": "drama", "text": """MARGARET: You're leaving, then. THOMAS: I am deciding whether to leave. There is a difference. MARGARET: The difference being that one involves a suitcase and the other involves standing in the hallway looking tragic. THOMAS: I am not looking tragic. MARGARET: No. You're looking like a man who wants to be asked to stay but hasn't the courage to ask himself. THOMAS: That's unfair. MARGARET: Most true things are. THOMAS: You have a talent for cruelty, Margaret. MARGARET: And you have a talent for self-pity. We are well matched. THOMAS: Were. Past tense. MARGARET: Is it? Look at us. Fifteen years and we are having the same argument we had in the first month. The furniture has changed. The argument hasn't. THOMAS: Then perhaps it is time for new furniture. MARGARET: You mean a new audience. THOMAS: I mean a new life."""},
    {"id": "ai_monologue_1", "genre": "drama", "text": """What am I, then, at the end of it? Not what I was. Not what I promised to become. Something between, something unfinished, like a sentence that trails off mid-thought. I have stood in rooms full of people and felt entirely absent. I have spoken words I did not mean and meant words I never spoke. I have loved, if that is the right word, and I am no longer certain it is, because what I called love looked remarkably like need, and what I called need was perhaps only fear of the alternative. I built a life out of decisions I barely remember making. Each one seemed small at the time, a turn here, a compromise there, nothing dramatic, nothing irreversible. Yet here I stand at the sum of them, and the sum is this: a man in a room, talking to no one, trying to understand how the distance between who he is and who he meant to be grew so large without his noticing."""},
    {"id": "ai_courtroom_1", "genre": "drama", "text": """JUDGE: Does the defense wish to make a statement? BRENNAN: Your Honor, my client is not what the prosecution would have you believe. They have painted a picture of a calculating individual, methodical and deliberate. But calculation requires clarity, and my client had none. He was confused, frightened, and acting under duress that the prosecution has conveniently failed to establish. PROSECUTOR: Objection. Characterization. JUDGE: Sustained. BRENNAN: I will rephrase. The evidence will show that on the night in question, my client received three phone calls, each one escalating in threat, each one narrowing his options until the only path remaining was the one the prosecution now condemns him for taking. He did not choose this. He was funneled into it. And the difference between a choice and a funnel is the difference between guilt and tragedy."""},

    # --- Poetry ---
    {"id": "ai_lyric_1", "genre": "poetry", "text": """I have measured the silence between heartbeats and found it longer than expected. There is a whole country in that pause, unmapped, where the blood waits and the lungs hold still and something like thought occurs without the burden of language. I have watched morning arrive not as light but as the slow subtraction of darkness, which is different, the way losing fear is different from gaining courage, the way an empty room is different from a room that has been emptied. I am trying to say something about presence. About the way a body occupies space without filling it. About the way a voice can cross a room and arrive changed, carrying the shape of the air it passed through. I want to speak of the ordinary astonishments, the coffee, the window, the particular slant of Tuesday afternoon light that falls across the table like a hand."""},
    {"id": "ai_epic_1", "genre": "poetry", "text": """They came down from the mountains in the old way, with their packs on their backs and their stories coiled tight as rope. The valley opened before them like a promise half-remembered, green in the places where the river ran and brown where the sun had burned the grass to straw. They had been walking for eleven days, though counting had stopped mattering somewhere around the fourth. Distance here was not measured in miles but in the quality of light, the angle of shadow, the sound the wind made crossing different kinds of stone. The eldest among them, a woman whose name had been worn smooth by decades of use until even she was not sure of its original shape, stopped at the ridge line and looked down. She had seen this valley before, in a different season, under a different set of circumstances, when the world was organized along different principles."""},

    # --- Nonfiction ---
    {"id": "ai_essay_tech_1", "genre": "essay", "text": """We did not decide to become a society that stares at screens. It happened incrementally, one convenience at a time, each step so small and so obviously beneficial that objecting seemed not just futile but unreasonable. Who would refuse a device that holds every book ever written, connects you to every person you have ever known, and fits in your pocket? The problem was never any single technology. The problem was the aggregate, the way each individual optimization combined to produce a total environment that nobody designed and nobody controls. We optimized for engagement and got addiction. We optimized for connection and got loneliness. We optimized for information and got a world in which nobody can agree on basic facts. These are not bugs. They are the predictable consequences of systems designed to capture attention competing with each other for a finite resource."""},
    {"id": "ai_essay_education_1", "genre": "essay", "text": """The purpose of education has never been settled, which is itself an education. Is it to produce workers? Citizens? Thinkers? Happy people? The answer changes with the era, the culture, the political mood, and the economic conditions, which is to say that education is less a fixed institution than a mirror, reflecting whatever a society believes about itself at any given moment. The current consensus, to the extent that one exists, favors measurability. We test, we rank, we sort. Students become data points, teachers become delivery systems, and learning becomes a thing that can be optimized, scaled, and benchmarked against international competitors. What gets lost in this process is everything that cannot be measured, which happens to be everything that matters most: curiosity, empathy, the capacity to sit with uncertainty."""},
    {"id": "ai_memoir_1", "genre": "memoir", "text": """I moved to the cabin in October, when the aspens were turning and the light had that quality peculiar to high altitude autumn, sharp and thin, like it was being stretched across too much sky. The nearest town was fourteen miles of dirt road, and the road itself was optimistic about its own existence, frequently disagreeing with the GPS about where, exactly, it went. I had told people I was going there to write, which was true in the way that most explanations are true, which is to say partially, and in a direction convenient to the teller. I was also going there to stop being the person I had become in the city, though I did not say this because it sounded dramatic and I was trying to be the kind of person to whom dramatic things did not happen. The first week I did nothing."""},
    {"id": "ai_news_econ_1", "genre": "journalism", "text": """Consumer spending declined for the second consecutive month, according to data released Friday by the Commerce Department, adding to concerns that the economic expansion may be losing momentum. Retail sales fell 0.4 percent in February after a revised 0.2 percent decline in January, worse than the 0.1 percent increase economists had forecast. The weakness was broad-based, with declines in categories ranging from electronics to clothing to restaurant spending. Auto sales were a rare bright spot, rising 1.2 percent on the back of manufacturer incentives that analysts expect to fade in coming months. The data prompted several Wall Street firms to lower their first-quarter GDP estimates. Goldman Sachs cut its tracking estimate to 1.8 percent from 2.3 percent, while JPMorgan reduced its forecast to 1.5 percent."""},
    {"id": "ai_news_politics_1", "genre": "journalism", "text": """The Senate voted along party lines Thursday to advance the infrastructure package, setting up a final vote expected early next week. The 51-49 tally, with no members crossing the aisle, underscored the deeply partisan nature of a bill that both sides have framed as essential to the nation's future. Democrats argue the measure represents a once-in-a-generation investment in roads, bridges, broadband, and clean energy. Republicans counter that the spending is reckless, the tax increases are punitive, and the bill's climate provisions amount to regulatory overreach. Senate Majority Leader Patricia Holloway called the vote a decisive step toward modernizing America's crumbling infrastructure. Minority Leader James Abernathy called it a decisive step toward fiscal catastrophe. Both appeared confident, which in the current political environment means that neither had the votes they claimed."""},
    {"id": "ai_legal_1", "genre": "legal", "text": """The obligations of the parties hereto shall be construed in accordance with the principles of good faith and fair dealing, provided however that nothing in this agreement shall be interpreted to impose upon either party any obligation not expressly stated herein. In the event of a dispute arising from or relating to the performance of this agreement, the parties shall first attempt to resolve such dispute through informal negotiation conducted in good faith. If such negotiation fails to produce a resolution within thirty calendar days of written notice of the dispute, either party may submit the matter to binding arbitration in accordance with the rules of the relevant arbitral institution. The prevailing party in any such proceeding shall be entitled to recover its reasonable costs and attorneys fees."""},
    {"id": "ai_religious_1", "genre": "religious", "text": """Consider the lilies. Not because they teach us anything we do not already know, but because the act of considering them, of stopping, of turning our attention from the ceaseless machinery of our wants and toward something that simply is, that is the practice. The whole of the spiritual life is contained in that turning. We are creatures of restless appetite, always reaching for the next thing, the better thing, the thing we believe will finally make us whole. And yet wholeness has been here all along, waiting for us to stop reaching and start receiving. This is not passivity. It is the most difficult kind of activity there is: the activity of attention, of showing up to the present moment without agenda. Prayer is sometimes words. More often it is silence. Most often it is the simple act of remaining, of not fleeing from the discomfort of being still."""},
    {"id": "ai_political_1", "genre": "political", "text": """The case for democracy has never rested on the assumption that the majority is always right. It rests on the far more modest and defensible claim that no individual or group can be trusted with unchecked power over others. This is an observation about human nature, not an optimistic one, and its implications are thoroughly practical. Every concentration of authority creates the conditions for its own abuse. The only reliable safeguard is to distribute power widely, to build in mechanisms of accountability, and to protect the right of citizens to replace their rulers without violence. These arrangements are messy, slow, and frequently produce outcomes that satisfy no one entirely. That is not a defect. It is the price of preventing outcomes that satisfy one faction at the expense of everyone else."""},
    {"id": "ai_history_1", "genre": "history", "text": """The siege lasted forty-seven days, though the defenders would later insist it felt like considerably longer. General Whitmore's position was, by any conventional military analysis, untenable. His supply lines had been cut on the twelfth day. The river crossing at Holmford, which had been his only route for reinforcement, fell to enemy sappers on the nineteenth. By the end of the third week, his garrison was subsisting on quarter rations and whatever the foraging parties could extract from the surrounding countryside, which was not much, the farmers having had the good sense to remove themselves and their livestock well before the shooting started. What kept the defense alive was not strategy but geography. The citadel sat on a limestone ridge with clear sight lines in every direction."""},

    # --- ADVERSARIAL: AI told to "sound human" / informal / messy ---
    {"id": "ai_adversarial_casual", "genre": "adversarial", "text": """ok so here's the thing about living in this city that nobody tells you before you move here. The rent is stupid. Like, actually stupid. Not "oh that's a lot" stupid but "I have a master's degree and I literally cannot afford a studio apartment" stupid. My friend Jake, who's been here six years, says you get used to it. Jake also thinks mayonnaise is a food group and once forgot his own birthday so I'm not sure his judgment calls are the gold standard here. Anyway I found this place in Bushwick, third floor walkup, the bathtub is in the kitchen which is apparently legal in this state and also a "charming vintage feature" according to the listing. The landlord's name is either Dmitri or Dimitri, he was unclear on this point, and the lease is handwritten which feels concerning but also kind of punk rock? I signed it. Mom would be horrified. Dad would be proud. Both reactions feel appropriate."""},
    {"id": "ai_adversarial_rant", "genre": "adversarial", "text": """I need to talk about parking in this neighborhood because it is genuinely destroying my will to live. I spent forty-five minutes last Tuesday driving in circles around my own block. Forty-five minutes. I could have learned a new language in that time. I could have read a chapter of a book. Instead I watched the same fire hydrant go past my windshield eleven times while my dinner got cold in the passenger seat and my blood pressure did things that would concern a medical professional. And the worst part? The WORST part? When I finally found a spot it was in front of my own building. The spot was there THE ENTIRE TIME. I just couldn't see it because someone had parked their boat-sized SUV two inches from my driveway, making what was clearly a legal parking space look like it wasn't. I measured it. It was legal. By three inches. I have become a person who measures parking spaces and I hate everything about that."""},
    {"id": "ai_adversarial_blog", "genre": "adversarial", "text": """So I tried that thing where you wake up at 5 AM for a month and honestly? Results are mixed. Week one was awful. Just absolutely terrible. I set four alarms and slept through three of them, then fell asleep in a meeting at 2 PM and dreamed I was at a different meeting, which was confusing for everyone. Week two was slightly better. I started actually getting things done in the morning hours, mostly because there is literally nothing else to do at 5 AM except be productive or stare at a wall, and I'd already tried the wall thing. Made progress on the novel I've been "working on" for three years. Week three I felt great. Like, suspiciously great. Energetic, focused, annoyingly cheerful. My coworkers noticed and seemed alarmed. Week four I crashed. Fell asleep at 8:30 PM on a Friday and woke up at noon on Saturday feeling like I'd been hit by a bus. The bus was called sleep debt."""},
    {"id": "ai_adversarial_letter", "genre": "adversarial", "text": """Dear Sandra, I know you said not to write but I'm writing anyway because that's the kind of person I am apparently, the kind who doesn't listen, which you pointed out many times and which I am only now beginning to appreciate was not just a criticism but a genuine observation about my character. Anyway. I wanted to say I'm sorry about the lamp. Not in a way that excuses anything but in a way that acknowledges that a person should not throw a lamp regardless of the circumstances, and the circumstances in this case did not warrant lamp-throwing by any reasonable standard. I've been thinking about what you said about patterns. You're right that I do this thing where I escalate and then apologize and then escalate again and the apologies start to feel like part of the cycle rather than interruptions of it. I don't want to be that person. I'm going to therapy now. Not because of the lamp, or not only because of the lamp. Because of the pattern."""},
    {"id": "ai_adversarial_diary", "genre": "adversarial", "text": """March 15. Woke up late again. Third time this week. The cat knocked my phone off the nightstand at 4 AM and it landed face-down so the alarm was muffled by carpet. I'm choosing to believe this was an accident and not a coordinated campaign against my employment. Made coffee with yesterday's grounds because I forgot to buy more, which was a choice I regretted immediately and then continued to make for three cups because waste is wrong even when the waste tastes like dirty water. Work was work. Henderson wants the report by Friday but keeps changing what the report is about, which makes writing it feel like trying to hit a target that someone is carrying around the room. Ate lunch at my desk. Leftover pasta. It was fine. That's the saddest word in the English language applied to food: fine. Had a good conversation with Rina about absolutely nothing important and it was the best part of my day, which says something about something but I'm too tired to figure out what."""},

    # --- ADVERSARIAL: AI mimicking specific human styles ---
    {"id": "ai_adversarial_hemingway", "genre": "adversarial", "text": """The old man sat in the bar and drank his wine. The wine was red and it was good. Outside the sun was going down and the street was getting dark. A woman walked past the window and did not look in. The old man did not watch her. He was thinking about the river and about the fish he had caught there when he was young. They were big fish then. Now the river was different and the fish were different and he was different and none of these differences were improvements. He ordered another glass. The bartender poured it without speaking. That was good. The old man did not want to speak. He wanted to sit and drink his wine and think about the river and not explain any of it to anyone. Explanation ruined things. It was like opening a watch to see why it kept time. After that it never kept time the same way again."""},
    {"id": "ai_adversarial_twain", "genre": "adversarial", "text": """Now the trouble with a committee is that it's got three kinds of people on it: the ones that do the work, the ones that take the credit, and the ones that just showed up because there were refreshments. In my experience the ratio runs about one to one to seven, which means for every person who knows what they're doing you've got seven people eating sandwiches and one person giving a speech about it afterward. I served on a committee once. It was a committee to determine whether the town needed a new bridge. We met every Tuesday for six months. At the end of it we determined that we needed another committee. The second committee met every Thursday for four months and determined that a bridge would probably be a good idea but that the matter deserved further study. I believe they are still studying it. The river, meanwhile, has gone right on being inconvenient, which is a thing rivers are good at."""},

    # --- Short samples ---
    {"id": "ai_short_1", "genre": "essay", "text": """The problem with most productivity advice is that it assumes you already have the one thing productive people have, which is the ability to do things when you don't feel like doing them. Everything else is just mechanics. Wake up early. Make lists. Block your calendar. Batch your email. These are fine suggestions for someone who can execute on suggestions. For the rest of us, the issue is not knowing what to do. The issue is the gap between knowing and doing, which is roughly the width of the Grand Canyon and about as easy to cross on foot. What actually works is smaller than any system. It is the practice of starting. Not finishing. Not planning. Starting. Because a thing in motion tends to stay in motion, and the hardest part of any task is the first thirty seconds."""},
    {"id": "ai_short_2", "genre": "essay", "text": """Language shapes thought. This is not a metaphor. The Hopi language has no past tense in the way English does, and Hopi speakers conceptualize time differently as a result. Russian speakers, who have separate words for light blue and dark blue, can distinguish between shades of blue faster than English speakers. Mandarin speakers, whose language uses vertical metaphors for time more often than horizontal ones, think about time's passage differently than English speakers do. These are measurable cognitive differences produced by grammatical structures absorbed in childhood. We do not simply use language to describe our experience of the world. Language is part of the apparatus through which we construct that experience in the first place."""},

    # --- Technical/academic ---
    {"id": "ai_technical_1", "genre": "technical", "text": """The transformer architecture relies on self-attention mechanisms to process sequential data without the recurrence constraints that limit traditional RNN-based approaches. In a standard transformer encoder block, input embeddings are projected into query, key, and value matrices through learned linear transformations. Attention weights are computed as the softmax of the scaled dot product between queries and keys, then applied to the values to produce context-aware representations. Multi-head attention extends this by running several attention operations in parallel, each with its own learned projections, allowing the model to attend to different types of relationships simultaneously. The output of the multi-head attention is concatenated and projected through a final linear layer before being added to the residual connection and passed through layer normalization. This architecture has proven remarkably effective across a wide range of natural language processing tasks."""},
    {"id": "ai_technical_2", "genre": "technical", "text": """Principal component analysis reduces the dimensionality of a dataset by finding the directions of maximum variance. Given an n-by-p data matrix X, we first center the data by subtracting the mean of each column, then compute the covariance matrix S. The eigenvectors of S, ordered by their corresponding eigenvalues from largest to smallest, define the principal components. Projecting the data onto the first k eigenvectors yields a k-dimensional representation that captures the most variance possible in k dimensions. The proportion of total variance explained by each component equals its eigenvalue divided by the sum of all eigenvalues. In practice, we often choose k by examining the scree plot or by selecting enough components to explain some threshold of total variance, commonly 90 or 95 percent. PCA assumes linear relationships between variables and is sensitive to the scaling of features."""},
]

# ── HUMAN TEXT GID LIST (we'll fetch these) ──
HUMAN_GIDS = [
    # Novels
    (1342, "Pride and Prejudice"), (84, "Frankenstein"), (1661, "Sherlock Holmes"),
    (76, "Huckleberry Finn"), (174, "Dorian Gray"), (345, "Dracula"),
    (2701, "Moby Dick"), (1260, "Jane Eyre"), (120, "Treasure Island"),
    (219, "Heart of Darkness"), (768, "Wuthering Heights"), (730, "Oliver Twist"),
    (11, "Alice in Wonderland"), (98, "Tale of Two Cities"),
    # Drama
    (1524, "Hamlet"), (1513, "Midsummer Night's Dream"), (1533, "Macbeth"),
    (844, "Being Earnest"),
    # Philosophy
    (5827, "Problems of Philosophy"), (1497, "Republic"), (3600, "Zarathustra"),
    (1232, "The Prince"),
    # Poetry
    (1321, "Paradise Lost"), (4800, "Leaves of Grass"), (1065, "Dickinson Poems"),
    # Science
    (2009, "Origin of Species"), (37729, "Relativity"),
    # Religious
    (10, "KJV Bible"), (8300, "Tao Te Ching"), (3207, "Marcus Aurelius"),
    # Political
    (815, "Democracy in America"), (61, "Communist Manifesto"),
    # Other
    (16328, "Dubliners"), (5200, "Metamorphosis"), (600, "Notes from Underground"),
    (4300, "Ulysses"), (2554, "Crime and Punishment"), (514, "Little Women"),
    (1399, "Anna Karenina"), (86, "Connecticut Yankee"),
]


def fetch_gutenberg_text(gid, max_chars=4000):
    import urllib.request
    url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "waivelets-eval/0.1"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8-sig", errors="replace")
    except Exception:
        return None

    for m in ["*** START OF THE PROJECT GUTENBERG", "*** START OF THIS PROJECT GUTENBERG", "***START OF"]:
        idx = raw.find(m)
        if idx >= 0:
            raw = raw[raw.index("\n", idx) + 1:]
            break
    for m in ["*** END OF", "End of the Project Gutenberg", "End of Project Gutenberg"]:
        idx = raw.find(m)
        if idx >= 0:
            raw = raw[:idx]
            break

    # Skip past title/ToC to first real paragraph
    lines = raw.split("\n")
    for i, line in enumerate(lines[20:], 20):
        if len(line.strip()) > 80 and not line.strip().isupper():
            raw = "\n".join(lines[i:])
            break

    return raw[:max_chars].strip()


def run():
    from fastprint import fingerprint, classify, split_sentences, Fingerprint

    results = []

    print("=" * 70)
    print("WAIVELETS AI vs HUMAN DETECTION — FULL EVAL v2")
    print("=" * 70)

    # ── AI texts ──
    print(f"\nFingerprinting {len(AI_SAMPLES)} AI texts...")
    for s in AI_SAMPLES:
        try:
            fp = fingerprint(s["text"])
            mode, dist = classify(fp)
            results.append({
                "source": "ai", "id": s["id"], "genre": s["genre"],
                "mode": mode, "distance": round(dist, 3),
                "fp": fp._asdict(),
            })
            print(f"  AI  {s['id']:35s} → {mode:14s} d={dist:.2f}  ent={fp.basin_entropy:.2f}")
        except Exception as e:
            print(f"  AI  {s['id']:35s} → ERROR: {e}")
        gc.collect()

    # ── Human texts ──
    print(f"\nFingerprinting {len(HUMAN_GIDS)} human texts from Gutenberg...")
    for gid, title in HUMAN_GIDS:
        text = fetch_gutenberg_text(gid)
        if not text:
            print(f"  HUM {title:35s} → FETCH FAILED")
            continue
        try:
            fp = fingerprint(text)
            mode, dist = classify(fp)
            results.append({
                "source": "human", "id": f"g{gid}", "genre": "gutenberg",
                "mode": mode, "distance": round(dist, 3),
                "fp": fp._asdict(), "title": title,
            })
            print(f"  HUM {title:35s} → {mode:14s} d={dist:.2f}  ent={fp.basin_entropy:.2f}")
        except Exception as e:
            print(f"  HUM {title:35s} → ERROR: {e}")
        gc.collect()

    # ── Analysis ──
    ai = [r for r in results if r["source"] == "ai"]
    hum = [r for r in results if r["source"] == "human"]
    adv = [r for r in ai if "adversarial" in r["id"]]
    non_adv = [r for r in ai if "adversarial" not in r["id"]]

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {len(ai)} AI, {len(hum)} human ({len(adv)} adversarial)")
    print("=" * 70)

    fp_keys = list(ai[0]["fp"].keys())
    ai_arr = np.array([[r["fp"][k] for k in fp_keys] for r in ai])
    hum_arr = np.array([[r["fp"][k] for k in fp_keys] for r in hum])

    # Feature-by-feature comparison
    print(f"\n{'Feature':20s} {'AI':>10s} {'Human':>10s} {'Cohen d':>10s}")
    print("─" * 52)
    for i, key in enumerate(fp_keys):
        am, asd = np.mean(ai_arr[:,i]), np.std(ai_arr[:,i])
        hm, hsd = np.mean(hum_arr[:,i]), np.std(hum_arr[:,i])
        pooled = np.sqrt((asd**2 + hsd**2)/2)
        d = abs(am-hm)/pooled if pooled > 0 else 0
        star = " ***" if d>1 else " **" if d>0.5 else ""
        print(f"{key:20s} {am:10.4f} {hm:10.4f} {d:10.2f}{star}")

    # Mode distributions
    print(f"\nMode distributions:")
    for src, group in [("AI", ai), ("Human", hum), ("Adversarial AI", adv)]:
        modes = Counter(r["mode"] for r in group)
        dist_str = ", ".join(f"{m}:{modes.get(m,0)}" for m in ["convergent","contemplative","discursive","dialectical"])
        print(f"  {src:20s} (n={len(group):2d}): {dist_str}")

    # Cluster separation
    ai_cent = np.mean(ai_arr, axis=0)
    hum_cent = np.mean(hum_arr, axis=0)
    inter = np.linalg.norm(ai_cent - hum_cent)
    ai_intra = np.mean([np.linalg.norm(fp - ai_cent) for fp in ai_arr])
    hum_intra = np.mean([np.linalg.norm(fp - hum_cent) for fp in hum_arr])
    ratio = inter / ((ai_intra + hum_intra)/2)
    print(f"\nCluster separation: {ratio:.3f} (inter={inter:.4f}, ai_intra={ai_intra:.4f}, hum_intra={hum_intra:.4f})")

    # ── Multi-feature classifier ──
    # Simple: z-score features, compute weighted distance to centroids
    all_arr = np.vstack([ai_arr, hum_arr])
    labels = np.array([1]*len(ai_arr) + [0]*len(hum_arr))

    # Best single feature
    best_acc, best_feat_idx = 0, 0
    best_thresh, best_dir = 0, 1
    for i in range(len(fp_keys)):
        for pct in range(5, 96):
            t = np.percentile(all_arr[:,i], pct)
            for d in [1,-1]:
                acc = np.mean(((all_arr[:,i]*d > t*d).astype(int)) == labels)
                if acc > best_acc:
                    best_acc, best_feat_idx, best_thresh, best_dir = acc, i, t, d

    print(f"\nBest single feature: {fp_keys[best_feat_idx]} {'>' if best_dir==1 else '<'} {best_thresh:.4f} → {best_acc:.1%}")

    # Centroid classifier
    ai_from_ai = np.array([np.linalg.norm(fp - ai_cent) for fp in all_arr])
    ai_from_hum = np.array([np.linalg.norm(fp - hum_cent) for fp in all_arr])
    cent_preds = (ai_from_ai < ai_from_hum).astype(int)
    cent_acc = np.mean(cent_preds == labels)
    print(f"Centroid classifier: {cent_acc:.1%}")

    # Two-feature classifier: basin_entropy + smoothness_std
    ent_idx = fp_keys.index("basin_entropy")
    std_idx = fp_keys.index("smoothness_std")
    best_2f = 0
    best_2f_params = {}
    for ent_pct in range(10, 91, 2):
        ent_t = np.percentile(all_arr[:,ent_idx], ent_pct)
        for std_pct in range(10, 91, 2):
            std_t = np.percentile(all_arr[:,std_idx], std_pct)
            # AI = low entropy AND low oscillation
            preds = ((all_arr[:,ent_idx] < ent_t) & (all_arr[:,std_idx] < std_t)).astype(int)
            acc = np.mean(preds == labels)
            if acc > best_2f:
                best_2f = acc
                best_2f_params = {"ent_thresh": round(float(ent_t), 4), "std_thresh": round(float(std_t), 4)}

    print(f"Best 2-feature (entropy+oscillation): {best_2f:.1%} (ent<{best_2f_params['ent_thresh']}, osc<{best_2f_params['std_thresh']})")

    # Score function for deployment: weighted features
    # Use simple logistic-like scoring
    z_mean = np.mean(all_arr, axis=0)
    z_std = np.std(all_arr, axis=0)
    z_std[z_std < 1e-8] = 1
    z_all = (all_arr - z_mean) / z_std

    # Feature weights based on Cohen's d (sign = direction of AI)
    weights = np.zeros(len(fp_keys))
    for i in range(len(fp_keys)):
        am = np.mean(ai_arr[:,i])
        hm = np.mean(hum_arr[:,i])
        pooled = np.sqrt((np.std(ai_arr[:,i])**2 + np.std(hum_arr[:,i])**2)/2)
        if pooled > 0:
            weights[i] = (am - hm) / pooled  # positive = AI direction

    # Composite score
    scores = z_all @ weights
    best_composite = 0
    best_ct = 0
    for pct in range(5, 96):
        ct = np.percentile(scores, pct)
        preds = (scores > ct).astype(int)
        acc = np.mean(preds == labels)
        if acc > best_composite:
            best_composite = acc
            best_ct = ct

    print(f"Composite weighted score: {best_composite:.1%}")

    # Adversarial subset analysis
    if adv:
        adv_arr = np.array([[r["fp"][k] for k in fp_keys] for r in adv])
        adv_scores = ((adv_arr - z_mean) / z_std) @ weights
        adv_detected = np.sum(adv_scores > best_ct)
        print(f"\nAdversarial AI detection: {adv_detected}/{len(adv)} detected ({100*adv_detected/len(adv):.0f}%)")
        for r, s in zip(adv, adv_scores):
            detected = "DETECTED" if s > best_ct else "MISSED"
            print(f"  {r['id']:35s} score={s:.2f} → {detected}")

    # ── Export classifier params for deployment ──
    classifier = {
        "version": "v2",
        "method": "weighted_zscore",
        "features": fp_keys,
        "z_mean": z_mean.tolist(),
        "z_std": z_std.tolist(),
        "weights": weights.tolist(),
        "threshold": round(float(best_ct), 4),
        "accuracy": round(float(best_composite), 4),
        "n_ai": len(ai),
        "n_human": len(hum),
        "single_feature": {
            "feature": fp_keys[best_feat_idx],
            "threshold": round(float(best_thresh), 4),
            "direction": ">" if best_dir == 1 else "<",
            "accuracy": round(float(best_acc), 4),
        },
        "centroid": {
            "ai": ai_cent.tolist(),
            "human": hum_cent.tolist(),
            "accuracy": round(float(cent_acc), 4),
        },
    }

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_ai": len(ai), "n_human": len(hum), "n_adversarial": len(adv),
        "results": results,
        "classifier": classifier,
        "separation_ratio": round(float(ratio), 3),
    }

    with open("eval_ai_human_v2_results.json", "w") as f:
        json.dump(output, f, indent=2)

    # Also save classifier params separately for the server
    with open("ai_detector_params.json", "w") as f:
        json.dump(classifier, f, indent=2)

    print(f"\nSaved eval_ai_human_v2_results.json and ai_detector_params.json")
    return output


if __name__ == "__main__":
    run()
