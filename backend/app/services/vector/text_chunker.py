class TextChunker:

    def chunk_text(
        self,
        text,
        chunk_size=600,
        overlap=120
    ):

        chunks = []

        start = 0

        while start < len(text):

            end = (
                start
                + chunk_size
            )

            chunks.append(
                text[start:end]
            )

            start += (
                chunk_size
                - overlap
            )

        return chunks