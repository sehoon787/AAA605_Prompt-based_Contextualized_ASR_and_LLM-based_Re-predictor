class PromptFormatter:
    def __init__(self, template=None):
        self.template = template or self.default_template()

    def default_template(self):
        return (
            "You are an ASR reranker.\n"
            "Prompt context: {utterance_prompt}\n"
            "Word context: {word_prompt}\n"
            "N-best hypotheses with acoustic model scores:\n"
            "{candidate_list}\n"
            "Choose the most likely correct transcription."
        )

    # TODO: word_prompt insert format WIP
    def format(self, utterance_prompt, word_prompt, candidates):
        candidate_list_str = ""
        for idx, cand in enumerate(candidates):
            candidate_list_str += f"{idx+1}. \"{cand['text']}\" (AM score: {cand['score']:.2f})\n"

        word_prompt_str = ""
        for idx, word in enumerate(word_prompt):
            word_prompt_str += f"{idx+1}. {word}\n"

        return self.template.format(
            utterance_prompt=utterance_prompt,
            word_prompt=word_prompt_str,
            candidate_list=candidate_list_str
        )
