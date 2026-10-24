from corpus_parser import create_parser

def main():
    parser = create_parser('brown', strip_headers=True)
    
    result = parser.parse('../data/corpora/brown/ca01')
    
    print(f"Paragraphs: {len(result)}")
    #print(f"Sentences {result['total_paragraphs']} paragraphs, {result['total_sentences']} sentences.")

    for item in result:
        print(f"--- Paragraph {item['paragraph_number']} ---")
        for j, sentence in enumerate(item['sentences'], start=1):
            print(f"  Sentence {j}: {sentence}")

if __name__ == '__main__':
    main()


    