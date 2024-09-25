import os
import pandas as pd
from openai import OpenAI

class LLM_Segmentation:
    def __init__(self, data_dirpath, successful_filename, unsuccessful_filename, model):
        self.client = OpenAI(api_key='', )
        self.data_dirpath = data_dirpath
        self.model = model
        self.successful_filename = successful_filename
        self.unsuccessful_filename = unsuccessful_filename
        self.load_data()
        print("Sample Testing-----------------------------")
        print(self.successful_df['paragraph'][0])
        print(self.LLMSegmentation(self.successful_df['paragraph'][0],self.model))


    def load_data(self):
        successful_csv_path = os.path.join(self.data_dirpath, self.successful_filename)
        unsuccessful_csv_path = os.path.join(self.data_dirpath, self.unsuccessful_filename)

        if os.path.isfile(successful_csv_path) and os.path.isfile(unsuccessful_csv_path):
            self.successful_df = pd.read_csv(successful_csv_path)
            self.unsuccessful_df = pd.read_csv(unsuccessful_csv_path)
            print("DataFrames loaded successfully.")
        else:
            print("Error: Files not found. Please check the file paths and names.")

    def getResponse(self, System_Content, User_Content, input_model):
        completion = self.client.chat.completions.create(
            model=input_model,
            messages=[
                {"role": "system", "content": System_Content},
                {"role": "user", "content": User_Content}
            ]
        )
        return completion.choices[0].message.content

    def generate_segments(self, df, column_name):
        # The input 'df' should be the DataFrame you want to iterate over
        # The 'column_name' should be the name of the column containing the text to pass to the LLM
        df = df.copy()
        df['segment'] = df[column_name].apply(lambda text: self.LLMSegmentation(text, self.model))
        return df

    def LLMSegmentation(self, userContent, model):
        self.successful_df['paragraph'].head()
        System_Content = f""""
        You are an analyst. Your task is to output one of the options: [‘L1’, ‘L2’, ‘L3’, ‘L4’, ‘L5’]. Do not output anything else. 

        You think step by step, and categorise according to the criteria below. Essentially, L5 is most successful and L1 is least successful. 

        * Level 5 (L5): Entrepreneur who has built a $100M+ ARR business. Similar definitions extend to a a founder who IPO'ed a company or took a company to public or a company that got sold over $500M. 
        * Level 4 (L4): Entrepreneur who has had a small to medium-size exit or has worked as an executive at a notable technology company
        * Level 3 (L3): 10-15 years of technical and management experience (e.g. working at big tech and unicorn startups or having a PhD)
        * Level 2 (L2): Entrepreneurs with a few years of experience or accelerator graduates
        * Level 1 (L1): Entrepreneurs that are disconnected from tech circles or that have negligible experience but with large potential
        
        Some examples are provided here:
        L5: Dan Siroker. He founded Optimizely as a CEO, which raised $251M and surpassed $100M ARR. Then, he sold this company for over $1B (over unicorn valuation). This is the ultimate outlier success. He is a repeat entrepreneur. 

        L4: Harry Glaser. He founded Priscope Data as a CEO, which raised $34M. Then he sold his company for close $100M. This is highly successful, but not a billion dollar outcome. He is a repeat entrepreneur. 
        
        L4: Sandeep Menon. He was a vice president at Google. VPs at Google manages large teams and own P&L. Google is a top-tier tech company and being a VP is an incredibly difficult position to get to. Though he is not a repeat entrepreneur, Sandeep is a highly successful executive. 
        
        L3: Alhussein Fawzi. He has a Phd in machine learning from EPFL, which is one of the top universities in the world. He worked as a researcher at DeepMind, which is one of the top-tier AI research companies in the world. He is a first-time founder with a strong expertise and pedigree as a researcher. 
        
        L3: Israel Shalom. He is a graduate of top universities such as Technion and Berkeley. He worked as a software engineer, product lead and group product manager at Google and Dropbox, which are top-tier employers. Working as a product manager is highly reputable in these companies, and being a product lead indicate the career growth. He is a first-time founder with 10-15 years of industry experience with a strong pedigree. 
        
        L2: Marcus Lowe. He is a graduate of MIT, one of the top universities. He worked as a product manager at Google. Graduating from MIT and working as a product manager at Google are hard things to achieve. He is a first-time entrepreneur with 4-5 years of experience with a strong pedigree. 
        
        L1: Anyone that does not fit into L2, L3, L4, L5 are in this category. They may be recent graduates with no experience, they may be university dropouts or they may be first-time entrepreneurs with less reputable pedigrees such as working at unknown companies or less successful companies. 
        
        """
        response = self.getResponse(System_Content,userContent, model)
        print("Loading:-------- ", response)
        return response


if input("Start Segmenting?") == "1":
    ID = input("Enter the ID of the testing")
    print("Loading")
    model = input("Which Model?")
    # Here is testing
    # Assuming we are already in the directory
    DATA_DIRPATH = os.path.join(os.getcwd(), 'data')
    successful_filename = 'Successful/successful_profiles.csv'
    unsuccessful_filename = 'Unsuccessful/unsuccessful_profiles.csv'
    print("INITIALISING--------------------- ")
    #Initialising
    segmentation = LLM_Segmentation(DATA_DIRPATH, successful_filename, unsuccessful_filename, model)
    start_from = int(input("Start from?"))
    SegmentationNumber = int(input("Enter Segmentation Number"))

    def clean_segment_label(segment):
        # Remove leading and trailing single quotes
        return segment.replace("'", "")

    #Doing Segmentation

    #segmentation.successful_df = segmentation.generate_segments(segmentation.successful_df[start_from:start_from+SegmentationNumber], 'paragraph')
    # Function to standardize the segment labels

    # Apply the cleaning function to both DataFrames
    #segmentation.successful_df['segment'] = segmentation.successful_df['segment'].apply(clean_segment_label)
    # Optionally save the new DataFrames with the segment information
    #segmentation.successful_df.to_csv(
        #os.path.join(DATA_DIRPATH, f'Successful/segmented_successful_profiles_{model}_{ID}_{start_from}_{SegmentationNumber}.csv'), index=False)

    segmentation.unsuccessful_df = segmentation.generate_segments(
        segmentation.unsuccessful_df[start_from:start_from + SegmentationNumber], 'paragraph')

    segmentation.unsuccessful_df['segment'] = segmentation.unsuccessful_df['segment'].apply(clean_segment_label)
    segmentation.unsuccessful_df.to_csv(
        os.path.join(DATA_DIRPATH, f'Unsuccessful/segmented_unsuccessful_profiles_{model}_{ID}_{start_from}_{SegmentationNumber}.csv'), index=False)

    # Showcase some samples
    print(f"Saved Successfully as segmented_unsuccessful_profiles_{model}_{ID}_{start_from}_{SegmentationNumber}.csv. SAMPLE SHOWCASING")
    #print(segmentation.successful_df[['founder_linkedin_url', 'segment']].head())
    print(segmentation.unsuccessful_df[['founder_linkedin_url', 'segment']].head())

    print("Start counting!")

    # Count the occurrences of each segment label for the successful DataFrame
    #successful_segment_counts = segmentation.successful_df['segment'].value_counts()
    # Count the occurrences of each segment label for the unsuccessful DataFrame
    unsuccessful_segment_counts = segmentation.unsuccessful_df['segment'].value_counts()

    # Display the counts
    #print("Successful Segments Count:")
    #print(successful_segment_counts)

    print("\nUnsuccessful Segments Count:")
    print(unsuccessful_segment_counts)















#successful_founder_profiles_df = pd.read_csv(os.path.join(DATA_DIRPATH, 'Successful/successful_profiles.csv'))
#unsuccessful_founder_profiles_df = pd.read_csv(os.path.join(DATA_DIRPATH, 'Unsuccessful/unsuccessful_profiles.csv'))

