base = 'https://f005.backblazeb2.com/file/yelp-review-data/chunk_'
end_of_file = '.csv'
CHUNKS_URLS = [base + str(num) + end_of_file for num in range(1,61)]