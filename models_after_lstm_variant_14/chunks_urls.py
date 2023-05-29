base = 'https://f005.backblazeb2.com/file/yelp-review-data-prepro/chunk_'
end_of_file = '.csv'
CHUNKS_URLS = [base + str(num) + end_of_file for num in range(1,61)]

sd_base = 'https://f005.backblazeb2.com/file/sarcasm-detection-preprocessed/sd_chunk_'
SD_CHUNKS_URLS = [sd_base + str(num) + end_of_file for num in range(1,11)]