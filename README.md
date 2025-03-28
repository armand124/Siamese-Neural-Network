# Task-Veridion
When I first read the task, I was very excited to try it. The first thing I did was think about how to retrieve the logos from the websites. Instantly, I came up with the solution of using Python along with Selenium to build a web scraper that extracts the logos and downloads them to my machine.

After that, I needed to design an algorithm that could successfully classify the images based on their similarity. This was an unusual challenge since there wasn’t a predefined dataset with labeled classes, making it very interesting to form a reliable set solely from the logos. So, what did I do? I attempted to group the extracted logos into the same folder whenever the website names showed a similarity above a certain threshold.

Once the dataset was reasonably well-structured, I had to decide on the classification algorithm. Since I was not allowed to use popular classification algorithms, I wanted to challenge myself by building something completely new to me. After a day or two of research, I decided to implement a Siamese Neural Network. This approach seemed like a great fit because it compares two images within the same neural network without modifying the weights, allowing me to train the model to assign similarity scores to pairs of images.

For feature extraction, I chose the pretrained ResNet50 model, given the short timeframe. However, I added my own touch by unfreezing the last layer to fine-tune the model, which significantly improved the results. Initially, the model’s performance was poor, so I experimented with different optimization techniques. I switched from the Adam optimizer to AdamW, adjusted the learning rate, and replaced the standard Sigmoid + Binary Cross-Entropy (BCE) loss with BCEWithLogitsLoss. These changes enhanced the model’s performance.

Ultimately, the model could reasonably predict logo similarity, but I had to verify the results manually because the dataset generator wasn’t entirely reliable and it still required human judgment.

Overall, the task was fun, and I learned a lot!
