from pathlib import Path
from typing import List
import chromadb
import pandas as pd
from langchain_core.documents import Document


def ingest_nutrition_excels(
  excel_dir: Path,
  engine: str = "openpyxl"
) -> List[Document]:
  """
  Load Nigerian/African nutrition Excel files and convert them into
  LangChain Documents suitable for semantic retrieval.

  - Embeds meaningful nutrition text in page_content
  - Keeps lightweight metadata for filtering/debugging
  """

  documents: List[Document] = []

  for excel_path in excel_dir.glob("*.xlsx"):
    df = pd.read_excel(excel_path, engine=engine)

    # Normalize column names (prevents silent KeyErrors)
    df.columns = [col.strip() for col in df.columns]

    for _, row in df.iterrows():
      try:
        content = f"""
          Food Name: {row.get('Food Name')}
          Local Name: {row.get('Local Name')}
          Scientific Name: {row.get('Scientific Name')}
          Category: {row.get('Category')}

          Calories per 100g: {row.get('Calories (per 100g)')} kcal
          Carbohydrates: {row.get('Carbs (g)')} g
          Protein: {row.get('Protein (g)')} g
          Fat: {row.get('Fat (g)')} g
          Fiber: {row.get('Fiber (g)')} g

          Micronutrients: {row.get('Micronutrients')}
          Glycemic Index: {row.get('Glycemic Index')}

          Health Benefits: {row.get('Health Benefits')}
          Health Risks: {row.get('Health Risks')}
          Medical Notes: {row.get('Medical Notes')}

          Primary Source: {row.get('Primary Source')}
          Reference Link: {row.get('Reference Link')}
          Secondary Authority: {row.get('Secondary Authority')}
          Confidence Level: {row.get('Confidence Level')}
          """.strip()

        documents.append(
          Document(
            page_content=content,
            metadata={
              "food_name": row.get("Food Name"),
              "local_name": row.get("Local Name"),
              "category": row.get("Category"),
              "confidence": row.get("Confidence Level"),
              "source": row.get("Primary Source"),
              "file": excel_path.name
            }
          )
        )

      except Exception as e:
        # Fail fast but visibly
        raise RuntimeError(
          f"Failed processing row in {excel_path.name}: {e}"
        )

  return documents




def test_retriever(retriever, text):
  docs = retriever.invoke(text)

  if not docs:
      print("⚠️ No documents retrieved")
  else:
      print(docs[0].page_content)
