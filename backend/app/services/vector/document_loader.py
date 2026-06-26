from pathlib import Path
from pypdf import PdfReader


class DocumentLoader:

    def load_directory(
        self,
        folder_path: str
    ):

        documents = []

        for pdf_file in Path(folder_path).glob("*.pdf"):

            print(
                f"Loading: {pdf_file.name}"
            )

            try:

                reader = PdfReader(
                    str(pdf_file)
                )

                for page_number, page in enumerate(
                    reader.pages,
                    start=1
                ):

                    page_text = page.extract_text()

                    if not page_text:
                        continue

                    page_text = page_text.strip()

                    if len(page_text) < 50:
                        continue

                    documents.append({

                        "source":
                            pdf_file.name,

                        "page":
                            page_number,

                        "text":
                            page_text
                    })


                page_count = len(reader.pages)

                print(
                    f"SUCCESS: {pdf_file.name} "
                    f"({page_count} pages)"
                )

            except Exception as e:

                print(
                    f"FAILED: {pdf_file.name}"
                )

                print(e)

        return documents