class PromptFormatter:
    def __init__(self, template=None):
        self.template = template or self.default_template()

    def default_template(self):
        return (
            "You are an ASR reranker.\n"
            "Prompt context: {utterance_prompt}\n"
            "Relevant keywords: {word_prompt}\n"
            "N-best hypotheses with acoustic model scores:\n"
            "{candidate_list}\n"
            "Choose the most likely correct transcription."
        )

    def format(self, utterance_prompt, word_prompt, candidates):
        candidate_list_str = ""
        for idx, cand in enumerate(candidates):
            candidate_list_str += f"{idx+1}. \"{cand['text']}\" (AM score: {cand['score']:.2f})\n"

        # 개선된 word_prompt 생성
        word_prompt_str = ", ".join(word_prompt)

        return self.template.format(
            utterance_prompt=utterance_prompt,
            word_prompt=word_prompt_str,
            candidate_list=candidate_list_str
        )
