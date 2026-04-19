from src.steps import ingest

def main():
    ingestion = ingest.Ingestion()
    ingestion.fetch_data()

if __name__ == "__main__":    main()