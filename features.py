import re
import nltk
import torch
import pandas as pd
import contractions
from preprocess_data import preprocess_text

##################################################################################
# All functions for manual detection
##################################################################################
def manual_features(tweets):
    def emoji_detector(tweet):
        """Check if a tweet contains emojis using regex."""
        emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"
                                    u"\U0001F300-\U0001F5FF"
                                    u"\U0001F680-\U0001F6FF"
                                    u"\U0001F700-\U0001F77F"
                                    u"\U000025A0-\U000025FF"
                                    u"\U000027B0-\U000027BF"
                                    u"\U0001F900-\U0001F9FF"
                                    u"\U00002600-\U000026FF"
                                    u"\U0001F300-\U0001F5FF"
                                    "]+", flags=re.UNICODE)
        return bool(re.search(emoji_pattern, tweet))

    def repeated_letters(word):
        return bool(re.search(r"(.)\1{2,}", word))  

    def contains_qmark_exclaim(tweet):
        return bool(re.search(r"[!?]", tweet))

    def repeated_punctuation(tweet):
        return bool(re.search(r"([!?])+([!?])+", tweet))

    def threedots(tweet):
        return bool(re.search(r"\.{3,}", tweet))

    def quotations(tweet):
        return bool(re.search(r"['\"]", tweet))

    def has_negation(sentence):
        """Check if a sentence contains any negation."""

        # List with negations taken from "Comparison of the Scope of Negation in Online News Articles" (2014)
        negations = {
            "hardly", "lack", "neither", "nor", 
            "never", "no", "nobody", "none", 
            "nothing", "nowhere", "not", "n't", 
            "cannot", "without", "bad", "uninspired", 
            "dxpensive", "disappoint", "ditch", "misunderstand" 
        }
        
        negation_pattern = re.compile(r"\b(?:" + "|".join(negations) + r")\b", flags=re.IGNORECASE)
        return bool(negation_pattern.search(sentence))

    def interjections(text):
        """
        Detect interjections in a text.
        Returns True if interjections are found, otherwise False.
        """
        # List of common interjections taken from https://www.enchantedlearning.com/wordlist/interjections.shtml
        interjections_list = ["aah", "ack", "agreed", "ah", "aha", "ahem", "alas", "all right", "amen", "argh", "as if", "aw", "ay", "aye", 
        "bah", "blast", "boo hoo", "bother", "boy", "brr", "by golly", "bye", "cheerio", "cheers", "chin up", "come on", "crikey", "curses", 
        "dear me", "doggone", "drat", "duh", "easy does it", "eek", "egads", "er", "exactly", "fair enough", "fiddle-dee-dee", "fiddlesticks", 
        "fie", "foo", "fooey", "gadzooks", "gah", "gangway", "g'day", "gee", "gee whiz", "geez", "gesundheit", "get lost", "get outta here", 
        "go on", "good", "good golly", "good job", "gosh", "gracious", "great", "grr", "gulp", "ha", "ha-ha", "hah", "hallelujah", "harrumph", 
        "haw", "hee", "here", "hey", "hmm", "ho hum", "hoo", "hooray", "hot dog", "how", "huh", "hum", "humbug", "hurray", "huzza", "I say", "ick", 
        "is it", "ixnay", "jeez", "just kidding", "just a sec", "just wondering", "kapish", "la", "la-di-dah", "lo", "look", "look here", "long time", 
        "lordy", "man", "meh", "mmm", "most certainly", "my", "my my", "my word", "nah", "naw", "never", "no", "no can do", "nooo", "not", "no thanks", 
        "no way", "nuts", "oh", "oho", "oh-oh", "oh no", "okay", "okey-dokey", "om", "oof", "ooh", "oopsey", "over", "oy", "oyez", "peace", "pff", "pew", 
        "phew", "pish posh", "psst", "ptui", "quite", "rah", "rats", "ready", "really","right", "right on", "roger", "roger that", "rumble", "say", "see ya",
        "shame", "shh", "shoo", "shucks", "sigh", "sleep tight", "snap", "sorry", "sssh", "sup", "sure", "ta", "ta-da", "ta ta", "take that", "tally ho", 
        "tch", "thanks", "there", "there there", "time out", "toodles", "touche", "tsk", "tsk-tsk", "tut", "tut-tut", "ugh", "uh", "uh-oh", "um", "ur", 
        "urgh", "very nice", "very well", "voila", "vroom", "wah", "well", "well done", "well, well", "what", "whatever", "whee", "when", "whoa", 
        "whoo", "whoopee", "whoops", "whoopsey", "whew", "why", "word", "wow", "wuzzup", "ya", "yea", "yeah", "yech", "yikes", "yippee", "yo", 
        "yoo-hoo", "you bet", "you don't say", "you know", "yow", "yum", "yummy", "zap", "zounds", "zowie", "zzz"]
        
        # Create a regex pattern to match any of the interjections
        interjection_pattern = re.compile(r"\b(?:" + "|".join(interjections_list) + r")\b", flags=re.IGNORECASE)
        
        # Search for interjections in the text
        return bool(interjection_pattern.search(text))


##################################################################################
# List with all functions.
###################################################################################

    functions_list = [
        emoji_detector,
        repeated_letters,
        contains_qmark_exclaim,
        repeated_punctuation,
        threedots,
        quotations,
        has_negation,
        interjections,
    ]

    ##################################################################################
    # Applying the functions to all tweets.
    ###################################################################################

    # Initialize a 2D NumPy array to store the results
    # Shape: (number of tweets, number of functions)
    results_tensor = torch.zeros((len(tweets), len(functions_list)), dtype=torch.int)

    # Process each tweet
    for j, tweet in enumerate(tweets):
        for i, function in enumerate(functions_list):
            # Apply the function and convert the result to 1 or 0
            result = 1 if function(tweet) else 0
            # Update the corresponding position in the matrix
            results_tensor[j, i] = result


    return results_tensor
