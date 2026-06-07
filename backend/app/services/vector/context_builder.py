class ContextBuilder:

    def build(
        self,
        retrieval_results
    ):

        documents = (
            retrieval_results[
                "documents"
            ][0]
        )

        metadatas = (
            retrieval_results[
                "metadatas"
            ][0]
        )
        distances = retrieval_results["distances"][0]
        print("DOCUMENT COUNT:")
        print(len(documents))

        print("METADATA COUNT:")
        print(len(metadatas))
        if metadatas:
            print("FIRST METADATA:")
            print(metadatas[0])
        filtered_docs = []
        ranked_docs = list(
            zip(
                documents,
                metadatas,
                distances
            )
        )

        ranked_docs.sort(
            key=lambda x: x[2]
        )
        sources = []

        seen_docs = set()

        MAX_DISTANCE = 1.0
        accepted_distances = []

        for doc, metadata, distance in ranked_docs:

            if distance > MAX_DISTANCE:
                continue

            if len(doc.split()) < 40:
                continue

            if "references" in doc.lower():
                continue

            if "bibliography" in doc.lower():
                continue

            if doc in seen_docs:
                continue

            accepted_distances.append(
                distance
            )

            seen_docs.add(doc)

            filtered_docs.append(doc)
            avg_distance = (
                sum(accepted_distances)
                / len(accepted_distances)
                if accepted_distances
                else 999
            )

            if avg_distance < 0.80:

                confidence = "high"

            elif avg_distance < 1.00:

                confidence = "medium"

            else:

                confidence = "low"
            source = (
                metadata.get(
                    "source"
                )
            )

            if (
                source
                and source
                not in sources
            ):
                sources.append(
                    source
                )

        context_parts = []

        for i, doc in enumerate(
            filtered_docs[:3]
        ):

            context_parts.append(
                f"""
            DOCUMENT {i+1}

            SOURCE:
            {metadata.get("source")}

            PAGE:
            {metadata.get("page")}

            CONTENT:
            {doc}
            """
            )

        context = "\n\n".join(
            context_parts
        )

        return {

            "context":
                context,

            "sources":
                sources,
            "confidence":
                confidence
        }