
import random
import pandas as pd
from pathlib import Path

random.seed(42)

# STYLE / NOISE HELPERS
PREFIXES = [
    "",
    "lowkey ",
    "honestly ",
    "ngl ",
    "tbh ",
    "bro ",
    "dude ",
    "wait ",
    "okay but ",
]

SUFFIXES = [
    "",
    " lol",
    " 😭",
    " tbh",
    " honestly",
    " fr",
    " :(",
    " <3",
    " haha",
    " ngl",
]

EMOJIS = [
    "",
    " 💀",
    " 😭",
    " ❤️",
    " 🫶",
    " 🙂",
    " 🥹",
    " 😭😭",
    " 😭💀",
]

PUNCT = ["", "!", "!!", "...", "."]


def distort(text):
    """Adds texting noise and formatting variation."""

    if random.random() < 0.3:
        text = text.lower()

    if random.random() < 0.1:
        text = text.upper()

    if random.random() < 0.25:
        text = text.replace("you", "u")

    if random.random() < 0.2:
        text = text.replace("your", "ur")

    if random.random() < 0.15:
        text = text.replace("really", "rly")

    if random.random() < 0.15:
        text = text.replace("to be honest", "tbh")

    return text.strip()


# PLATONIC

PLATONIC_OPENERS = [
    "want to grab lunch",
    "you free this weekend",
    "thanks for helping me",
    "youre such a good friend",
    "bro youre hilarious",
    "proud of you",
    "hope youre doing okay",
    "lets hang out soon",
    "you always make me laugh",
    "thanks for always being there",
    "youre the realest person i know",
    "game night soon",
    "miss the whole group",
    "you killed that presentation",
    "you got this",
    "thanks for listening to me",
    "youre like family to me",
    "study session later",
    "send me the notes pls",
    "hope work wasnt too stressful",
    "glad were friends",
    "youre my favorite idiot",
    "love u bro",
    "text me when u get home",
    "the group isnt the same without u",
]

PLATONIC_CARING = [
    "my friend always checks if i got home safe",
    "hes protective of all his friends",
    "she always makes sure everyone is okay",
    "hes just a caring person",
    "good friends look out for each other",
]

PLATONIC_LOVE = [
    "i love my friends",
    "love you bro",
    "i love hanging out with them",
    "my friends mean everything to me",
    "theyre my support system",
    "i care about my friends deeply",
    "i love my best friend platonically",
    "my friends are family to me",
]
PLATONIC_ENDINGS = [
    "the whole crew should come",
    "lets invite everyone",
    "as friends obviously",
    "bestie energy",
    "ur genuinely awesome",
    "you always help me out",
    "thanks again",
    "youre such a real one",
    "cant wait to catch up",
    "hope ur proud of yourself",
    "you deserve good things",
    "we need another group trip",
]

PLATONIC_SEED = [
    "Hey, want to grab lunch tomorrow? Just the usual crew.",
    "friendships take effort from both sides",
    "every relationship needs communication, even friendships",
    "healthy friendships require consistency",
    "close friends should make time for each other too",
    "platonic relationships still need emotional effort",
    "even best friends drift apart without effort",
    "friendship is still a relationship at the end of the day",
    "Thanks for helping me move last weekend, you're such a good friend.",
    "Let me know if you need notes from class, happy to share.",
    "Bro, that was the funniest thing I've seen all week.",
    "Can't believe you remembered my sister's birthday, you're the best!",
    "We should do a study session this weekend if you're free.",
    "Miss hanging out with the old gang, we should plan something.",
    "Your advice about the job interview really helped, thank you.",
    "I'll always have your back, that's what friends are for.",
    "Let me know when you're free — game night at mine?",
    "You always know how to cheer me up, honestly.",
    "Glad we're in the same project group, this is going to be fun.",
    "You're like the sibling I never had, seriously.",
    "Just checking in to see how you're doing after everything.",
    "We should catch that movie together, as a group thing.",
    "Thanks for always being honest with me, even when it's hard.",
    "Our friendship means a lot to me, I hope you know that.",
    "Let's plan a road trip with everyone this summer!",
    "You're such a good listener, I really appreciate you.",
    "Want to hit the gym together this week? Back to the grind.",
    "I'll bring snacks if you set up the board game.",
    "Text me when you get home safe, okay?",
    "Working with you on this project is always a blast.",
    "Just saw something that reminded me of our inside joke lol.",
    "You're genuinely one of the funniest people I know.",
    "Hope you feel better soon, let me know if you need anything.",
    "I'll always vouch for you, don't even worry.",
    "Our friendship is one of the best things in my life.",
    "You should apply — I'll be your reference no problem.",
    "Hadn't heard from you in a while, just making sure you're okay.",
    "I know you'll crush it tomorrow, you've got this.",
    "We need to hang out more, it's been too long.",
    "You never have to go through that alone — I'm here.",
    "Let's make a plan and actually stick to it this time.",
    "Hope today was kinder to you than yesterday.",
    "You're the person I always want in my corner.",
    "Can I borrow that book when you're done? It sounds great.",
    "I was telling my roommate how much you make me laugh.",
    "You never have to face that alone, I'm always here.",
    "Dude your presentation today was actually amazing.",
    "Haha remember when we tried to cook that and nearly burned the place down?",
    "I told the whole group what you did — we're all so proud of you!",
    "I told my mom about you, she thinks you sound like a great person.",
    "You're one of the most genuine people I know.",
    "I mentioned you to my coworker — you two have a lot in common.",
    "You're always there when I need someone to vent to.",
    "Proud of how far you've come this year.",
    "Group chat has been dead, we need to revive it.",
    "You and I have been friends for so long I forget life before you.",
]

# ROMANTIC

ROMANTIC_OPENERS = [
    "i cant stop thinking about you",
    "i miss you already",
    "you looked beautiful today",
    "i think im falling for you",
    "i really like you",
    "you make my heart race",
    "i wish you were here",
    "i want to hold your hand",
    "youre always on my mind",
    "i think about you constantly",
    "you make me nervous in a good way",
    "i want more than friendship",
    "youre special to me",
    "i love talking to you",
    "youre the first person i think about",
    "i wish i was next to you right now",
    "you make ordinary days feel special",
    "i want to see you all the time",
    "youre literally my favorite person",
    "i adore you",
    "i think i caught feelings",
    "i want to kiss you",
    "i feel safe with you",
    "i cant explain how much i like you",
    "you mean everything to me",
]

ROMANTIC_ENDINGS = [
    "and its terrifying",
    "if im being honest",
    "i needed to tell you",
    "please dont laugh",
    "ive never felt like this before",
    "youre all i think about lately",
    "i hope you feel the same",
    "youre stuck in my head",
    "i cant hide it anymore",
    "you make me so happy",
    "just us sounds perfect",
    "you genuinely make my day better",
]

ROMANTIC_SEED = [
    "I can't stop thinking about you since we talked last night.",
    "he said he loves me",
    "he said he wants to marry me",
    "he asked me to be his girlfriend",
    "he confessed his feelings",
    "he said he wants a future with me",
    "he told me hes in love with me",
    "Every time I see your name on my phone I smile like an idiot.",
    "I really like you and I think you should know that.",
    "You're the most beautiful person I've ever met, inside and out.",
    "Would you want to go on a date with me? Just the two of us.",
    "I've been wanting to tell you this for a while — I have feelings for you.",
    "Falling asleep thinking about you has become my routine.",
    "I don't want to just be friends anymore, if I'm being honest.",
    "You make me feel something I haven't felt in a really long time.",
    "I'd love to take you somewhere special this weekend.",
    "I think about our conversations all day, is that weird?",
    "When I picture my future, somehow you're always in it.",
    "You looked absolutely stunning tonight, I couldn't take my eyes off you.",
    "I'm jealous of anyone who gets to spend more time with you than me.",
    "Being around you feels like home.",
    "I want to hold your hand and never let go.",
    "There's no one I'd rather spend a quiet evening with.",
    "I missed you the moment you left.",
    "You have no idea how often you cross my mind.",
    "I've never connected with someone the way I connect with you.",
    "I want to be the reason you smile today.",
    "I like you more than I probably should.",
    "Can I take you out for dinner sometime? Just us.",
    "You're the first person I want to call when something happens.",
    "I think I'm falling for you, and honestly it scares me a little.",
    "Being with you feels effortless in the best possible way.",
    "I want to learn everything about you.",
    "My heart literally races when I see you.",
    "Every song I hear lately reminds me of you.",
    "I want to make you feel as special as you make me feel.",
    "I keep re-reading our messages, is that too much?",
    "You're the first thing I think about in the morning.",
    "I'd give up a lot just to spend one more hour with you.",
    "You're everything I didn't know I was looking for.",
    "I think I'm in love with you.",
    "Date me? Please? Okay I'll stop being awkward but seriously.",
    "The way you laugh makes the whole world better.",
    "I find myself making excuses just to see you.",
    "You deserve someone who adores you — and I want to be that person.",
    "Being honest: I want more than just friendship with you.",
    "I wish I could kiss you goodnight.",
    "You make ordinary moments feel extraordinary.",
    "I've never wanted to impress someone as much as I want to impress you.",
    "You're the exception to every rule I set for myself.",
    "There's something about you that just draws me in.",
    "I feel lucky every time you choose to spend time with me.",
    "I want to be yours, if you'll have me.",
    "You're the person I didn't know I needed.",
    "I keep hoping you feel the same way.",
    "Waking up next to you would make any morning perfect.",
]

ESCALATION_ROMANTIC = [

    "he confessed his feelings",
    "he admitted he likes me",
    "he said he has feelings for me",
    "he told me he wants more than friendship",
    "he said he sees me differently now",
    "he asked if id date him",
]

SARCASM_ROMANTIC = [

    "bro im so cooked",
    "hes ruining my life bc i like him so much",
    "totally dont have feelings for him lol",
    "yeah im DEFINITELY not obsessed with him",
    "bro got me giggling and kicking my feet",
    "why do i smile every time he texts me",
    "im trying so hard not to fall for him",
    "nah bc why is he always on my mind",
    "this man has me losing my sanity",
    "i hate him (i like him)",
]
# AMBIGUOUS

AMBIG_OPENERS = [
    "you feel different to me",
    "i think about you a lot",
    "im not sure what this is",
    "i feel weird around you lately",
    "you make me nervous",
    "i feel close to you",
    "i miss you even when we just talked",
    "you matter to me more than you know",
    "i dont know where the line is anymore",
    "theres something between us",
    "i feel safe around you",
    "you understand me differently",
    "i keep replaying our conversations",
    "youre important to me",
    "i cant fully explain this feeling",
    "sometimes i wonder about us",
    "being around you feels different",
    "i like talking to you more than anyone else",
    "youre the person i always reach for",
    "you make me overthink everything",
    "im confused about my feelings",
    "i feel attached to you",
    "youre constantly on my mind",
    "i dont want to lose whatever this is",
]

AMBIG_ENDINGS = [
    "and idk why",
    "maybe its nothing",
    "or maybe im overthinking",
    "its confusing honestly",
    "do you ever feel that too",
    "i cant tell what this means",
    "maybe its just me",
    "i dont know how to explain it",
    "its hard to describe",
    "and thats kinda scary",
    "but i like it",
    "its weird in a good way",
]

AMBIGUOUS_SEED = [
    "I always look forward to seeing you, more than anyone else.",
    "You make everything better just by being there.",
    "I think about you a lot, hope that's okay.",
    "There's something about talking to you that I just can't explain.",
    "I feel really comfortable around you, more than most people.",
    "You're different from everyone else in my life.",
    "I got you something small — I just thought of you when I saw it.",
    "I don't know what I'd do without you honestly.",
    "I stay up late just because I don't want to stop talking to you.",
    "Being around you just feels right.",
    "You're the first person I wanted to tell.",
    "I'd drop everything for you, you know that right?",
    "I notice when you're not yourself.",
    "I'm not sure what this is, but I like it.",
    "You mean more to me than I know how to say.",
    "I keep thinking about that conversation we had.",
    "I feel like we have something really special.",
    "You're always on my mind, no matter what's going on.",
    "I want to understand everything you're going through.",
    "There's no one I trust more than you.",
    "Is it normal to feel this close to someone you haven't known that long?",
    "I feel like you get me in a way no one else does.",
    "Every time I'm with you I don't want it to end.",
    "I'm not sure what we are but I know I want you in my life.",
    "I've never opened up to anyone the way I open up to you.",
    "You have a way of making me feel seen.",
    "Sometimes I wonder if you feel what I feel.",
    "I miss you even when we just talked an hour ago.",
    "I care about you more than I can put into words.",
    "You're the only person I want to talk to right now.",
    "There's a version of this that scares me and a version that excites me.",
    "I don't really understand what's happening between us but I like it.",
    "Do you ever wonder what things would be like if we were different?",
    "I feel something when I'm with you that I can't quite name.",
    "I could talk to you forever and still not say everything I want to.",
    "You're the person I reach for when things fall apart.",
    "I think about the things you say long after you've said them.",
    "I'm not sure where the line is between us anymore.",
    "Something feels different lately and I can't tell if it's me or us.",
    "I would do a lot for you, probably more than I should.",
    "I want to be close to you but I'm not sure what that means.",
    "You give me butterflies and I'm still figuring out why.",
    "When you hug me I don't want to let go.",
    "I didn't expect to feel this way about you.",
    "I'd rather be anywhere with you than somewhere perfect alone.",
    "You make me think about things I usually don't think about.",
    "I catch myself hoping our plans never get cancelled.",
    "You're important to me in a way I haven't fully figured out.",
    "I feel nervous and comfortable at the same time when I'm around you.",
    "I like who I am when I'm with you.",
]
# 
# SARCASM
# 

SARCASTIC = [
    ("yeah sure im TOTALLY not obsessed with u", 1, "Romantic"),
    ("wow thanks for ruining my life bestie", 0, "Platonic"),
    ("haha imagine if i actually had feelings for u", 2, "Ambiguous"),
    ("im kidding... unless?", 2, "Ambiguous"),
    ("bro youre literally the worst ❤️", 0, "Platonic"),
]

# 
# NEGATIVE / NEUTRAL
# 

NEUTRAL = [
    "the package arrived yesterday",
    "meeting moved to 4pm",
    "my wifi stopped working",
    "submit the pdf before midnight",
    "the weather is horrible today",
]

# 
# SOCIAL MEDIA STYLE
# 

SOCIAL_STYLE = [
    ("bro got me giggling n shi 😭", 1, "Romantic"),
    ("lowkey attached ngl", 2, "Ambiguous"),
    ("why am i smiling at my phone rn", 1, "Romantic"),
    ("they got me acting DIFFERENT", 2, "Ambiguous"),
    ("im cooked", 2, "Ambiguous"),
    ("need them biblically", 1, "Romantic"),
    ("friendship or yearning call it", 2, "Ambiguous"),
]

# HARD BORDERLINE

HARD_BORDERLINE = [
    ("goodnight ❤️", 2, "Ambiguous"),
    ("i love talking to you", 2, "Ambiguous"),
    ("youre my comfort person", 2, "Ambiguous"),
    ("you make me feel safe", 2, "Ambiguous"),
    ("wish you were here", 2, "Ambiguous"),
]

# MULTI TURN CONVERSATIONS

MULTI_TURN = [
    (
        [
            "did u get home safe",
            "yeah just now",
            "good i was worried about u",
        ],
        1,
        "Romantic",
    ),
    (
        [
            "who's bringing snacks",
            "ill bring chips",
            "bet see u guys at 7",
        ],
        0,
        "Platonic",
    ),
    (
        [
            "youve been acting different lately",
            "different how",
            "idk just closer",
        ],
        2,
        "Ambiguous",
    ),
]

# TEMPLATES

TEMPLATES = [
    "{prefix}{opener}{punct}",
    "{prefix}{opener} {ending}{punct}",
    "{prefix}{opener}{emoji}",
    "{prefix}{opener} {ending}{emoji}",
    "{prefix}{opener} {suffix}",
    "{prefix}{opener} {ending} {suffix}",
]


# GENERATOR

def generate_samples(openers, endings, label_name, label_id, target_n=1500):

    unique_texts = set()

    while len(unique_texts) < target_n:

        opener = random.choice(openers)
        ending = random.choice(endings)
        template = random.choice(TEMPLATES)

        text = template.format(
            prefix=random.choice(PREFIXES),
            opener=opener,
            ending=ending,
            suffix=random.choice(SUFFIXES),
            emoji=random.choice(EMOJIS),
            punct=random.choice(PUNCT),
        )

        text = distort(text)
        text = " ".join(text.split())

        unique_texts.add(text)

    rows = []

    for text in unique_texts:
        rows.append({
            "text": text,
            "label": label_id,
            "label_name": label_name,
        })

    return rows

# BUILD DATASET

def build_dataset(output_path="data/dataset.csv"):

    records = []

    # Main generated classes
    records += generate_samples(
        PLATONIC_OPENERS,
        PLATONIC_ENDINGS,
        "Platonic",
        0,
        target_n=1500,
    )

    records += generate_samples(
        ROMANTIC_OPENERS,
        ROMANTIC_ENDINGS,
        "Romantic",
        1,
        target_n=1500,
    )

    records += generate_samples(
        AMBIG_OPENERS,
        AMBIG_ENDINGS,
        "Ambiguous",
        2,
        target_n=1500,
    )
    
    # Add handcrafted seed examples
    from itertools import chain

    for text in chain(PLATONIC_SEED, PLATONIC_CARING, PLATONIC_LOVE):
        records.append({"text": text, "label": 0, "label_name": "Platonic"})

    for text in chain(ROMANTIC_SEED, ESCALATION_ROMANTIC, SARCASM_ROMANTIC):
        records.append({"text": text, "label": 1, "label_name": "Romantic"})

    for text in AMBIGUOUS_SEED:
        records.append({"text": text, "label": 2, "label_name": "Ambiguous"})
    # Sarcastic
    for text, label, label_name in SARCASTIC:
        records.append({
            "text": text,
            "label": label,
            "label_name": label_name,
        })

    # Social style
    for text, label, label_name in SOCIAL_STYLE:
        records.append({
            "text": text,
            "label": label,
            "label_name": label_name,
        })

    # Borderline
    for text, label, label_name in HARD_BORDERLINE:
        records.append({
            "text": text,
            "label": label,
            "label_name": label_name,
        })

    # Multi-turn
    for conv, label, label_name in MULTI_TURN:
        joined = " [SEP] ".join(conv)

        records.append({
            "text": joined,
            "label": label,
            "label_name": label_name,
        })

    # Neutral mixed lightly into platonic
    for text in NEUTRAL:
        records.append({
            "text": text,
            "label": 0,
            "label_name": "Platonic",
        })

    # Build dataframe
    df = pd.DataFrame(records)

    # Remove duplicates
    df = df.drop_duplicates(subset=["text"])

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    print(f"Dataset size: {len(df)}")
    print(df.label_name.value_counts())

    return df


if __name__ == "__main__":
    build_dataset()


def build_dataset_v2(output_path="data/dataset.csv"):
    """
    Extends build_dataset() with targeted hard-case examples
    that address failures found during evaluation:
      - "love my friends" false positives
      - Short romantic confessions ("He said he loves me")
      - Future/commitment language ("wants a future with me")
      - Direct date asks ("asking if I'd ever date him")
      - Social perception ambiguity ("people think we're dating")
      - Safe-checking ambiguity ("checks if I got home safe")
      - GenZ sarcasm ("bro im so cooked")
    """
    import pandas as pd
    from pathlib import Path

    base_df = build_dataset(output_path=None)

    extras = [
        # Platonic false-positive fixes
        ("I love my friends so much, they mean everything to me.",          0, "Platonic"),
        ("My friends are literally my emotional support system.",            0, "Platonic"),
        ("I genuinely just love all my friends.",                            0, "Platonic"),
        ("I love hanging out with my whole friend group.",                   0, "Platonic"),
        ("I love them all so much, they are my people.",                    0, "Platonic"),
        ("They are my emotional anchors, my chosen family.",                0, "Platonic"),
        ("I don't know what I'd do without my friend group.",               0, "Platonic"),
        ("We've been friends for years, basically family at this point.",    0, "Platonic"),
        ("She's one of the few people I really trust with anything.",        0, "Platonic"),
        ("We've supported each other through literally everything.",         0, "Platonic"),
        ("He's basically like a brother to me at this point.",               0, "Platonic"),
        ("She's just my best friend, nothing more.",                         0, "Platonic"),
        ("A lot of good friends check in on each other like that.",          0, "Platonic"),
        ("We support each other, that's what good friendships look like.",   0, "Platonic"),
        ("Honestly my friends are everything to me, platonically.",          0, "Platonic"),
        ("People always think we're dating but we're literally just close friends.", 0, "Platonic"),

        # Short romantic confessions
        ("He said he loves me.",                                             1, "Romantic"),
        ("She said she loves me.",                                           1, "Romantic"),
        ("He told me he loves me.",                                          1, "Romantic"),
        ("I think I love him too.",                                          1, "Romantic"),
        ("And honestly I think I love him too.",                             1, "Romantic"),
        ("I realized I love her.",                                           1, "Romantic"),
        ("I think I love her too honestly.",                                 1, "Romantic"),
        ("He told me he likes me.",                                          1, "Romantic"),
        ("She told me she has feelings for me.",                             1, "Romantic"),
        ("He confessed he likes me yesterday.",                              1, "Romantic"),
        ("Yesterday he confessed he has feelings for me.",                   1, "Romantic"),
        ("He said he sees me as more than a friend.",                        1, "Romantic"),
        ("She told me she's caught feelings for me.",                        1, "Romantic"),
        ("She admitted she's liked me for months.",                          1, "Romantic"),
        ("He said he's falling for me.",                                     1, "Romantic"),

        # Future / commitment language 
        ("And now he says he wants a future with me.",                       1, "Romantic"),
        ("He told me he wants a future with me.",                            1, "Romantic"),
        ("She said she sees a future with me.",                              1, "Romantic"),
        ("He said he wants to build something with me.",                     1, "Romantic"),
        ("He wants to make it official.",                                    1, "Romantic"),
        ("She said she wants us to be together.",                            1, "Romantic"),
        ("He said he wants to be with me.",                                  1, "Romantic"),

        # Direct date asks 
        ("He's asking if I'd ever date him.",                                1, "Romantic"),
        ("And now he's asking if I'd ever date him.",                        1, "Romantic"),
        ("She asked if I'd ever consider dating her.",                       1, "Romantic"),
        ("He asked me out directly.",                                        1, "Romantic"),
        ("He literally asked me out today.",                                 1, "Romantic"),
        ("She asked if we could be more than friends.",                      1, "Romantic"),
        ("He joked about marrying me someday.",                              1, "Romantic"),

        # Smile / nervous / texting signals 
        ("I smile every time he texts me and I can't help it.",              1, "Romantic"),
        ("Why do I get nervous every time he texts me?",                     1, "Romantic"),
        ("I can't stop thinking about him no matter what I do.",             1, "Romantic"),
        ("He told me he thinks about me all the time.",                      1, "Romantic"),
        ("I light up every time I see his name on my phone.",                1, "Romantic"),
        ("like why do i smile every time he texts me",                       1, "Romantic"),
        ("why do i smile every time she texts me this is embarrassing",      1, "Romantic"),

        #  Ambiguous: safe-checking 
        ("He always checks if I got home safe.",                             2, "Ambiguous"),
        ("She always makes sure I got home okay.",                           2, "Ambiguous"),
        ("He texts to make sure I got home safe every time.",                2, "Ambiguous"),
        ("He always checks up on me when I'm out.",                          2, "Ambiguous"),

        #  Ambiguous: social perception 
        ("People always think we're dating lol.",                            2, "Ambiguous"),
        ("Everyone thinks we're a couple but we're not.",                    2, "Ambiguous"),
        ("People keep asking if we're dating which is weird.",               2, "Ambiguous"),
        ("My friends keep saying we act like a couple.",                     2, "Ambiguous"),

        #  Ambiguous: uncertainty after platonic context 
        ("I miss him when we don't talk, it feels weird.",                   2, "Ambiguous"),
        ("I feel different around him lately and I don't know why.",         2, "Ambiguous"),
        ("I don't know what to do with these feelings.",                     2, "Ambiguous"),
        ("Now I don't know how to act around him.",                          2, "Ambiguous"),
        ("Everything feels different since he said that.",                   2, "Ambiguous"),
        ("I can't tell if what I'm feeling is more than friendship.",        2, "Ambiguous"),
        ("I miss him more than I'd miss a regular friend.",                  2, "Ambiguous"),

        #  Ambiguous: GenZ sarcasm / confused tone 
        ("bro he is literally ruining my life and I don't even know why.",   2, "Ambiguous"),
        ("bro hes literally ruining my life 😭",                             2, "Ambiguous"),
        ("im so cooked I can't stop thinking about them.",                   2, "Ambiguous"),
        ("I'm literally so done, why do I care this much.",                  2, "Ambiguous"),
        ("why am I like this, I keep thinking about them.",                  2, "Ambiguous"),
        ("he's ruining my life and I hate that I love it.",                  2, "Ambiguous"),
        ("I'm cooked, totally cooked.",                                      2, "Ambiguous"),
        ("but no I totally do not have feelings for him right?",             2, "Ambiguous"),
        ("I keep telling myself it's nothing but idk.",                      2, "Ambiguous"),
        ("We're just friends but sometimes it feels like more.",             2, "Ambiguous"),
        ("I'm probably overthinking this but something feels off.",          2, "Ambiguous"),
        ("Something changed between us and I can't explain it.",             2, "Ambiguous"),

        # "We talk every day" should be Platonic not Ambiguous 
        ("We talk basically every day now.",                                0, "Platonic"),
        ("We text each other every single day.",                           0, "Platonic"),
        ("We've been talking every day for months.",                       0, "Platonic"),
        ("We call each other every day, we're really close.",              0, "Platonic"),

        # "just my best friend" should be Platonic 
        ("But honestly she's just my best friend.",                        0, "Platonic"),
        ("He's genuinely just my best friend, nothing more.",              0, "Platonic"),
        ("She's just a friend to me, truly.",                              0, "Platonic"),
        ("He's my closest friend but that's all it is.",                   0, "Platonic"),
        ("Honestly, she's just my best friend and I love that.",           0, "Platonic"),

        # ── Fix: "People think we're dating" should be Ambiguous 
        ("People always think we're dating lol.",                          2, "Ambiguous"),
        ("Everyone thinks we're a couple but we're really not.",           2, "Ambiguous"),
        ("People keep asking if we're dating, it's so weird.",             2, "Ambiguous"),
        ("My friends all assume we're together.",                          2, "Ambiguous"),

        #  "checks if I got home safe" should be Ambiguous 
        ("He always checks if I got home safe.",                           2, "Ambiguous"),
        ("She texts me whenever I'm out late to make sure I'm okay.",      2, "Ambiguous"),
        ("He always makes sure I got home safe after we hang out.",        2, "Ambiguous"),
        ("She checks up on me every time I go out.",                       2, "Ambiguous"),

        # "im so cooked" should be Ambiguous 
        ("im so cooked i cant stop thinking about him",                    2, "Ambiguous"),
        ("im so cooked, im literally catching feelings",                   2, "Ambiguous"),
        ("bro im so cooked for real",                                      2, "Ambiguous"),
        ("im cooked fr i like him so much",                                2, "Ambiguous"),

        #  "sometimes I wonder if he feels the same" should be Ambiguous
        ("Sometimes I wonder if he feels the same way.",                   2, "Ambiguous"),
        ("I wonder if she feels the same way I do.",                       2, "Ambiguous"),
        ("Sometimes I catch myself wondering if he likes me too.",         2, "Ambiguous"),
        ("I often wonder if she has feelings for me too.",                 2, "Ambiguous"),

        #  "don't know if friend sees me romantically" = Ambiguous 
        ("I don't know if my friend sees me as a romantic interest.",      2, "Ambiguous"),
        ("I don't know if he likes me romantically or just as a friend.",  2, "Ambiguous"),
        ("Not sure if she sees me as more than a friend.",                 2, "Ambiguous"),
        ("I wonder if he thinks of me as just a friend or something more.",2, "Ambiguous"),
        ("Does he see me only as a friend or could he have feelings?",     2, "Ambiguous"),
    ]

    import pandas as pd
    extra_df = pd.DataFrame(extras, columns=["text", "label", "label_name"])
    df = pd.concat([base_df, extra_df], ignore_index=True).drop_duplicates("text").reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    print(f"Dataset v2: {len(df)} samples")
    print(df.label_name.value_counts().to_string())
    return df
