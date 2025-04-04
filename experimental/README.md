# BPMN Element Template Suggester

## Problem Statement

When modeling business processes using [BPMN.io](http://bpmn.io/), users often need to apply specific element templates to BPMN elements to extend their functionality with custom properties or behaviors. However, manually selecting the most appropriate template can be time-consuming, especially when many templates are available. This project introduces a Python backend solution that powers an extension for [BPMN.io](http://bpmn.io/), adding a "Show suggestions" button to the context pad of BPMN elements. When clicked, this button analyzes the selected element's name and type to suggest relevant element templates, streamlining the modeling process.

## Primary Solution: Cosine Similarity Approach

### Theory and Approach

The primary solution, implemented in `app_using_cosine_similarity.py`, uses a two-step process to suggest element templates:

1. **Filtering by Element Type:**
    - The system filters a predefined list of templates (loaded from `all_elements.json`) to include only those applicable to the selected element's type (e.g., `"bpmn:Task"`). This ensures that suggestions are relevant to the element's structural context within the BPMN model.
2. **Semantic Similarity with Cosine Similarity:**
    - For the filtered templates, the system computes semantic similarity between the element's name (e.g., "Send email") and the combined `name` and `description` fields of each template.
    - A pre-trained SentenceTransformer model (`multi-qa-mpnet-base-dot-v1`) encodes both the element's name and the template texts into high-dimensional vectors.
    - Cosine similarity is calculated between the element's name vector and each template's vector.
    - The top 5 templates with the highest similarity scores are returned as suggestions.

This approach leverages the transformer model's ability to capture semantic meaning, ensuring that suggestions are contextually relevant. The embeddings for all templates are precomputed at startup for efficiency, and HTML tags are stripped from text inputs using BeautifulSoup to focus on raw content.

## Alternate Solutions

The project includes several variations of the primary approach, summarized in the table below:

| **Approach** | **Description** | **File** |
| --- | --- | --- |
| Dot Product Similarity | Replaces cosine similarity with dot product for potentially faster computation, though less normalized. | `app_using_dot_product.py` |
| TF-IDF Based Similarity | Uses TF-IDF vectors instead of transformer embeddings; may not capture semantic meaning as effectively. | `app_using_tfidf.py` |
| Expanded Information | Includes additional template details (e.g., group labels, property labels/descriptions) in the embedding text for richer context. | `expanded_info_app.py` |

These alternate solutions explore different trade-offs in speed, accuracy, and context awareness.

## Instructions to Run the Code

Follow these steps to set up and run the application:

1. **Install Dependencies:**
    - Ensure you have Python 3.10 or higher installed.
    - Install the required packages using pip:
        
        ```
        pip install -r experimental/requirements.txt
        
        ```
        
    - No additional environment variables need to be configured.
2. **Run the FastAPI Application:**
    - Navigate to the directory containing the desired application file (e.g., `app_using_cosine_similarity.py`).
    - Launch the server with uvicorn:
        
        ```
        uvicorn app_using_cosine_similarity:app --host 0.0.0.0 --port 8000
        
        ```
        
    - Replace `app_using_cosine_similarity` with the appropriate module name (e.g., `app_using_dot_product`) to run an alternate solution.
3. **Access the API:**
    - The API will be available at `http://localhost:8000`.
        

The application assumes the presence of an `all_elements.json` file in the same directory, containing the template data.

## Table of Results


| Test Case ID | Input Name | Input Type | Suggestion Rank | Suggestion ID | Suggestion Name | Similarity Score |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Send email | bpmn:Task | 1 | io.camunda.connectors.SendGrid.v2 | SendGrid Outbound Connector | 0.617240846157074 |
| 1 | Send email | bpmn:Task | 2 | io.camunda.connectors.MSTeams.v1 | Microsoft Teams Outbound Connector | 0.49730828404426575 |
| 1 | Send email | bpmn:Task | 3 | io.camunda.connectors.MSFT.O365.Mail.v1 | Microsoft Office 365 Mail Connector | 0.4485277831554413 |
| 1 | Send email | bpmn:Task | 4 | io.camunda.connectors.AWSSQS.v1 | Amazon SQS Outbound Connector | 0.4445931017398834 |
| 1 | Send email | bpmn:Task | 5 | io.camunda.connectors.WhatsApp.v1 | WhatsApp Business Outbound Connector | 0.434861958026886 |
| 2 | Send message | bpmn:Task | 1 | io.camunda.connectors.SendGrid.v2 | SendGrid Outbound Connector | 0.5727589130401611 |
| 2 | Send message | bpmn:Task | 2 | io.camunda.connectors.RabbitMQ.v1 | RabbitMQ Outbound Connector | 0.5520817637443542 |
| 2 | Send message | bpmn:Task | 3 | io.camunda.connectors.MSTeams.v1 | Microsoft Teams Outbound Connector | 0.5431579351425171 |
| 2 | Send message | bpmn:Task | 4 | io.camunda.connectors.Slack.v1 | Slack Outbound Connector | 0.5349791049957275 |
| 2 | Send message | bpmn:Task | 5 | io.camunda.connectors.AWSSNS.v1 | Amazon SNS Outbound connector | 0.5228978991508484 |
| 3 | Setup project | bpmn:Task | 1 | io.camunda.connectors.Asana.v1 | Asana Outbound Connector | 0.476737380027771 |
| 3 | Setup project | bpmn:Task | 2 | io.camunda.connectors.GitLab.v1 | GitLab Outbound Connector | 0.38987892866134644 |
| 3 | Setup project | bpmn:Task | 3 | io.camunda.connectors.MSTeams.v1 | Microsoft Teams Outbound Connector | 0.3808334171772003 |
| 3 | Setup project | bpmn:Task | 4 | io.camunda.connectors.GitHub.v1 | GitHub Outbound Connector | 0.375124454498291 |
| 3 | Setup project | bpmn:Task | 5 | io.camunda.connectors.GoogleDrive.v1 | Google Drive Outbound Connector | 0.33285677433013916 |
| 4 | Call REST API | bpmn:Task | 1 | io.camunda.connectors.HttpJson.v2 | REST Outbound Connector | 0.7207006812095642 |
| 4 | Call REST API | bpmn:Task | 2 | io.camunda.connectors.Salesforce.v1 | Salesforce Outbound Connector | 0.5641088485717773 |
| 4 | Call REST API | bpmn:Task | 3 | io.camunda.connectors.CamundaOperate.v1 | Camunda Operate Outbound connector | 0.474442720413208 |
| 4 | Call REST API | bpmn:Task | 4 | io.camunda.connectors.Twilio.v1 | Twilio Outbound Connector | 0.42708075046539307 |
| 4 | Call REST API | bpmn:Task | 5 | io.camunda.connectors.AWSLAMBDA.v2 | AWS Lambda Outbound Connector | 0.41143035888671875 |
| 5 | Get distance from Google Maps | bpmn:Task | 1 | io.camunda.connectors.GoogleMapsPlatform.v1 | Google Maps Platform Outbound Connector | 0.5353685021400452 |
| 5 | Get distance from Google Maps | bpmn:Task | 2 | io.camunda.connectors.GoogleSheets.v1 | Google Sheets Outbound Connector | 0.34131526947021484 |
| 5 | Get distance from Google Maps | bpmn:Task | 3 | io.camunda.connectors.GoogleDrive.v1 | Google Drive Outbound Connector | 0.3014565706253052 |
| 5 | Get distance from Google Maps | bpmn:Task | 4 | io.camunda.connectors.CamundaOperate.v1 | Camunda Operate Outbound connector | 0.2987685203552246 |
| 5 | Get distance from Google Maps | bpmn:Task | 5 | io.camunda.connectors.BluePrism.v1 | Blue Prism Outbound Connector | 0.2865719795227051 |
| 6 | Send message on Slack | bpmn:Task | 1 | io.camunda.connectors.Slack.v1 | Slack Outbound Connector | 0.7338482737541199 |
| 6 | Send message on Slack | bpmn:Task | 2 | io.camunda.connectors.RabbitMQ.v1 | RabbitMQ Outbound Connector | 0.5740988850593567 |
| 6 | Send message on Slack | bpmn:Task | 3 | io.camunda.connectors.MSTeams.v1 | Microsoft Teams Outbound Connector | 0.5098016262054443 |
| 6 | Send message on Slack | bpmn:Task | 4 | io.camunda.connectors.AWSSQS.v1 | Amazon SQS Outbound Connector | 0.5090879201889038 |
| 6 | Send message on Slack | bpmn:Task | 5 | io.camunda.connectors.WhatsApp.v1 | WhatsApp Business Outbound Connector | 0.5043389201164246 |
| 7 | Receive order | bpmn:StartEvent | 1 | io.camunda.connectors.AWSSQS.startmessage.v1 | Amazon SQS Message Start Event Connector | 0.4327721893787384 |
| 7 | Receive order | bpmn:StartEvent | 2 | io.camunda.connectors.AWSSQS.StartEvent.v1 | Amazon SQS Start Event Connector | 0.4128342866897583 |
| 7 | Receive order | bpmn:StartEvent | 3 | io.camunda.connectors.AWSEventBridge.MessageStart.v1 | Amazon EventBridge Message Start Event Connector | 0.3933475613594055 |
| 7 | Receive order | bpmn:StartEvent | 4 | io.camunda.connectors.inbound.AWSSNS.MessageStartEvent.v1 | SNS HTTPS Message Start Event Connector Subscription | 0.382180392742157 |
| 7 | Receive order | bpmn:StartEvent | 5 | io.camunda.connectors.inbound.RabbitMQ.MessageStart.v1 | RabbitMQ Message Start Event Connector | 0.3796226978302002 |
| 8 | Wait for approval | bpmn:IntermediateCatchEvent | 1 | io.camunda.connectors.AWSSQS.intermediate.v1 | Amazon SQS Intermediate Message Catch Event connector | 0.39577537775039673 |
| 8 | Wait for approval | bpmn:IntermediateCatchEvent | 2 | io.camunda.connectors.inbound.KafkaIntermediate.v1 | Kafka Intermediate Catch Event Connector | 0.3226417303085327 |
| 8 | Wait for approval | bpmn:IntermediateCatchEvent | 3 | io.camunda.connectors.inbound.RabbitMQ.Intermediate.v1 | RabbitMQ Intermediate Catch Event Connector | 0.3056662678718567 |
| 8 | Wait for approval | bpmn:IntermediateCatchEvent | 4 | io.camunda.connectors.webhook.GithubWebhookConnectorIntermediate.v1 | GitHub Webhook Intermediate Catch Event Connector | 0.30365124344825745 |
| 8 | Wait for approval | bpmn:IntermediateCatchEvent | 5 | io.camunda.connectors.inbound.AWSSNS.IntermediateCatchEvent.v1 | SNS HTTPS Intermediate Catch Event Connector | 0.2991527318954468 |
| 9 | GitHub issue | bpmn:BoundaryEvent | 1 | io.camunda.connectors.webhook.GithubWebhookConnectorBoundary.v1 | GitHub Webhook Boundary Event Connector | 0.5477398633956909 |
| 9 | GitHub issue | bpmn:BoundaryEvent | 2 | io.camunda.connectors.inbound.Slack.BoundaryEvent.v1 | Slack Webhook Boundary Event Connector | 0.2684527337551117 |
| 9 | GitHub issue | bpmn:BoundaryEvent | 3 | io.camunda.connectors.AWSSQS.boundary.v1 | Amazon SQS Boundary Event Connector | 0.22071051597595215 |
| 9 | GitHub issue | bpmn:BoundaryEvent | 4 | io.camunda.connectors.webhook.WebhookConnectorBoundary.v1 | Webhook Boundary Event Connector | 0.2194419652223587 |
| 9 | GitHub issue | bpmn:BoundaryEvent | 5 | io.camunda.connectors.inbound.RabbitMQ.Boundary.v1 | RabbitMQ Boundary Event Connector | 0.20743246376514435 |
| 10 | Create Slack channel | bpmn:Task | 1 | io.camunda.connectors.Slack.v1 | Slack Outbound Connector | 0.7483879923820496 |
| 10 | Create Slack channel | bpmn:Task | 2 | io.camunda.connectors.RabbitMQ.v1 | RabbitMQ Outbound Connector | 0.464194655418396 |
| 10 | Create Slack channel | bpmn:Task | 3 | io.camunda.connectors.MSTeams.v1 | Microsoft Teams Outbound Connector | 0.4159455895423889 |
| 10 | Create Slack channel | bpmn:Task | 4 | io.camunda.connectors.WhatsApp.v1 | WhatsApp Business Outbound Connector | 0.40201234817504883 |
| 10 | Create Slack channel | bpmn:Task | 5 | io.camunda.connectors.GitHub.v1 | GitHub Outbound Connector | 0.39947420358657837 |
| 11 | Email sender | bpmn:Task | 1 | io.camunda.connectors.SendGrid.v2 | SendGrid Outbound Connector | 0.5077352523803711 |
| 11 | Email sender | bpmn:Task | 2 | io.camunda.connectors.MSFT.O365.Mail.v1 | Microsoft Office 365 Mail Connector | 0.4030872583389282 |
| 11 | Email sender | bpmn:Task | 3 | io.camunda.connectors.Slack.v1 | Slack Outbound Connector | 0.40002119541168213 |
| 11 | Email sender | bpmn:Task | 4 | io.camunda.connectors.EasyPost.v1 | Easy Post Outbound Connector | 0.3959598243236542 |
| 11 | Email sender | bpmn:Task | 5 | io.camunda.connectors.MSTeams.v1 | Microsoft Teams Outbound Connector | 0.36821603775024414 |
| 12 | Order received | bpmn:StartEvent | 1 | io.camunda.connectors.AWSSQS.startmessage.v1 | Amazon SQS Message Start Event Connector | 0.4060399830341339 |
| 12 | Order received | bpmn:StartEvent | 2 | io.camunda.connectors.AWSSQS.StartEvent.v1 | Amazon SQS Start Event Connector | 0.39680516719818115 |
| 12 | Order received | bpmn:StartEvent | 3 | io.camunda.connectors.inbound.RabbitMQ.MessageStart.v1 | RabbitMQ Message Start Event Connector | 0.3641442060470581 |
| 12 | Order received | bpmn:StartEvent | 4 | io.camunda.connectors.inbound.AWSSNS.MessageStartEvent.v1 | SNS HTTPS Message Start Event Connector Subscription | 0.3538869619369507 |
| 12 | Order received | bpmn:StartEvent | 5 | io.camunda.connectors.AWSEventBridge.MessageStart.v1 | Amazon EventBridge Message Start Event Connector | 0.3483712375164032 |
| 13 | Approval wait | bpmn:IntermediateCatchEvent | 1 | io.camunda.connectors.AWSSQS.intermediate.v1 | Amazon SQS Intermediate Message Catch Event connector | 0.3962556719779968 |
| 13 | Approval wait | bpmn:IntermediateCatchEvent | 2 | io.camunda.connectors.inbound.KafkaIntermediate.v1 | Kafka Intermediate Catch Event Connector | 0.3216707706451416 |
| 13 | Approval wait | bpmn:IntermediateCatchEvent | 3 | io.camunda.connectors.inbound.AWSSNS.IntermediateCatchEvent.v1 | SNS HTTPS Intermediate Catch Event Connector | 0.29145389795303345 |
| 13 | Approval wait | bpmn:IntermediateCatchEvent | 4 | io.camunda.connectors.webhook.GithubWebhookConnectorIntermediate.v1 | GitHub Webhook Intermediate Catch Event Connector | 0.28182223439216614 |
| 13 | Approval wait | bpmn:IntermediateCatchEvent | 5 | io.camunda.connectors.http.Polling | HTTP Polling Intermediate Catch Event Connector | 0.2787559926509857 |
| 14 | Notification sender | bpmn:IntermediateThrowEvent | 1 | io.camunda.connectors.AWSSQS.intermediate.v1 | Amazon SQS Intermediate Message Catch Event connector | 0.4389360845088959 |
| 14 | Notification sender | bpmn:IntermediateThrowEvent | 2 | io.camunda.connectors.inbound.Slack.IntermediateCatchEvent.v1 | Slack Webhook Intermediate Catch Event Connector | 0.3948357105255127 |
| 14 | Notification sender | bpmn:IntermediateThrowEvent | 3 | io.camunda.connectors.inbound.RabbitMQ.Intermediate.v1 | RabbitMQ Intermediate Catch Event Connector | 0.3824406862258911 |
| 14 | Notification sender | bpmn:IntermediateThrowEvent | 4 | io.camunda.connectors.Twilio.Webhook.Intermediate.v1 | Twilio Intermediate Catch Event Connector | 0.3801603615283966 |
| 14 | Notification sender | bpmn:IntermediateThrowEvent | 5 | io.camunda.connectors.webhook.GithubWebhookConnectorIntermediate.v1 | GitHub Webhook Intermediate Catch Event Connector | 0.3556925654411316 |
| 15 | Send email to customer | bpmn:Task | 1 | io.camunda.connectors.SendGrid.v2 | SendGrid Outbound Connector | 0.5404134392738342 |
| 15 | Send email to customer | bpmn:Task | 2 | io.camunda.connectors.AWSSQS.v1 | Amazon SQS Outbound Connector | 0.4459289312362671 |
| 15 | Send email to customer | bpmn:Task | 3 | io.camunda.connectors.MSTeams.v1 | Microsoft Teams Outbound Connector | 0.4435942471027374 |
| 15 | Send email to customer | bpmn:Task | 4 | io.camunda.connectors.AWSSNS.v1 | Amazon SNS Outbound connector | 0.42805108428001404 |
| 15 | Send email to customer | bpmn:Task | 5 | io.camunda.connectors.MSFT.O365.Mail.v1 | Microsoft Office 365 Mail Connector | 0.4271038770675659 |
| 16 | Make API request | bpmn:Task | 1 | io.camunda.connectors.HttpJson.v2 | REST Outbound Connector | 0.6121540665626526 |
| 16 | Make API request | bpmn:Task | 2 | io.camunda.connectors.Salesforce.v1 | Salesforce Outbound Connector | 0.5595124959945679 |
| 16 | Make API request | bpmn:Task | 3 | io.camunda.connectors.CamundaOperate.v1 | Camunda Operate Outbound connector | 0.4991230368614197 |
| 16 | Make API request | bpmn:Task | 4 | io.camunda.connectors.AutomationAnywhere | Automation Anywhere Outbound Connector | 0.4207152724266052 |
| 16 | Make API request | bpmn:Task | 5 | io.camunda.connectors.Twilio.v1 | Twilio Outbound Connector | 0.41809597611427307 |
| 17 | Email | bpmn:IntermediateCatchEvent | 1 | io.camunda.connectors.AWSSQS.intermediate.v1 | Amazon SQS Intermediate Message Catch Event connector | 0.29721665382385254 |
| 17 | Email | bpmn:IntermediateCatchEvent | 2 | io.camunda.connectors.inbound.Slack.IntermediateCatchEvent.v1 | Slack Webhook Intermediate Catch Event Connector | 0.25563517212867737 |
| 17 | Email | bpmn:IntermediateCatchEvent | 3 | io.camunda.connectors.webhook.GithubWebhookConnectorIntermediate.v1 | GitHub Webhook Intermediate Catch Event Connector | 0.25271469354629517 |
| 17 | Email | bpmn:IntermediateCatchEvent | 4 | io.camunda.connectors.inbound.RabbitMQ.Intermediate.v1 | RabbitMQ Intermediate Catch Event Connector | 0.24283862113952637 |
| 17 | Email | bpmn:IntermediateCatchEvent | 5 | io.camunda.connectors.Twilio.Webhook.Intermediate.v1 | Twilio Intermediate Catch Event Connector | 0.23281490802764893 |
| 18 | Receive Email | bpmn:IntermediateCatchEvent | 1 | io.camunda.connectors.AWSSQS.intermediate.v1 | Amazon SQS Intermediate Message Catch Event connector | 0.4352796971797943 |
| 18 | Receive Email | bpmn:IntermediateCatchEvent | 2 | io.camunda.connectors.Twilio.Webhook.Intermediate.v1 | Twilio Intermediate Catch Event Connector | 0.4164913594722748 |
| 18 | Receive Email | bpmn:IntermediateCatchEvent | 3 | io.camunda.connectors.inbound.RabbitMQ.Intermediate.v1 | RabbitMQ Intermediate Catch Event Connector | 0.4135434031486511 |
| 18 | Receive Email | bpmn:IntermediateCatchEvent | 4 | io.camunda.connectors.inbound.Slack.IntermediateCatchEvent.v1 | Slack Webhook Intermediate Catch Event Connector | 0.38280993700027466 |
| 18 | Receive Email | bpmn:IntermediateCatchEvent | 5 | io.camunda.connectors.AWSEventBridge.intermediate.v1 | Amazon EventBridge Intermediate Catch Event Connector | 0.38166987895965576 |
| 19 | Approval | bpmn:Task | 1 | io.camunda.connectors.AutomationAnywhere | Automation Anywhere Outbound Connector | 0.36950260400772095 |
| 19 | Approval | bpmn:Task | 2 | io.camunda.connectors.EasyPost.v1 | Easy Post Outbound Connector | 0.34989339113235474 |
| 19 | Approval | bpmn:Task | 3 | io.camunda.connectors.AWSSQS.v1 | Amazon SQS Outbound Connector | 0.3345377445220947 |
| 19 | Approval | bpmn:Task | 4 | io.camunda.connectors.UIPath.v1 | UiPath Outbound Connector | 0.3264550268650055 |
| 19 | Approval | bpmn:Task | 5 | io.camunda.connectors.KAFKA.v1 | Kafka Outbound Connector | 0.3129385709762573 |
| 20 | Approval | bpmn:IntermediateCatchEvent | 1 | io.camunda.connectors.AWSSQS.intermediate.v1 | Amazon SQS Intermediate Message Catch Event connector | 0.3016926348209381 |
| 20 | Approval | bpmn:IntermediateCatchEvent | 2 | io.camunda.connectors.inbound.KafkaIntermediate.v1 | Kafka Intermediate Catch Event Connector | 0.2668655514717102 |
| 20 | Approval | bpmn:IntermediateCatchEvent | 3 | io.camunda.connectors.inbound.RabbitMQ.Intermediate.v1 | RabbitMQ Intermediate Catch Event Connector | 0.234971821308136 |
| 20 | Approval | bpmn:IntermediateCatchEvent | 4 | io.camunda.connectors.inbound.AWSSNS.IntermediateCatchEvent.v1 | SNS HTTPS Intermediate Catch Event Connector | 0.23304596543312073 |
| 20 | Approval | bpmn:IntermediateCatchEvent | 5 | io.camunda.connectors.webhook.GithubWebhookConnectorIntermediate.v1 | GitHub Webhook Intermediate Catch Event Connector | 0.2323189079761505 |
| 21 | Send Email Notification | bpmn:Task | 1 | io.camunda.connectors.SendGrid.v2 | SendGrid Outbound Connector | 0.5942775011062622 |
| 21 | Send Email Notification | bpmn:Task | 2 | io.camunda.connectors.MSTeams.v1 | Microsoft Teams Outbound Connector | 0.4917500615119934 |
| 21 | Send Email Notification | bpmn:Task | 3 | io.camunda.connectors.AWSSQS.v1 | Amazon SQS Outbound Connector | 0.4829312562942505 |
| 21 | Send Email Notification | bpmn:Task | 4 | io.camunda.connectors.AWSSNS.v1 | Amazon SNS Outbound connector | 0.4701760411262512 |
| 21 | Send Email Notification | bpmn:Task | 5 | io.camunda.connectors.MSFT.O365.Mail.v1 | Microsoft Office 365 Mail Connector | 0.4539983868598938 |
| 22 | Fire John | bpmn:Task | 1 | io.camunda.connectors.Slack.v1 | Slack Outbound Connector | 0.3201301097869873 |
| 22 | Fire John | bpmn:Task | 2 | io.camunda.connectors.HttpJson.v2 | REST Outbound Connector | 0.28826743364334106 |
| 22 | Fire John | bpmn:Task | 3 | io.camunda.connectors.MSTeams.v1 | Microsoft Teams Outbound Connector | 0.27162760496139526 |
| 22 | Fire John | bpmn:Task | 4 | io.camunda.connectors.SendGrid.v2 | SendGrid Outbound Connector | 0.2681942582130432 |
| 22 | Fire John | bpmn:Task | 5 | io.camunda.connectors.UIPath.v1 | UiPath Outbound Connector | 0.26360124349594116 |
| 23 | SEND EMAIL | bpmn:Task | 1 | io.camunda.connectors.SendGrid.v2 | SendGrid Outbound Connector | 0.617240846157074 |
| 23 | SEND EMAIL | bpmn:Task | 2 | io.camunda.connectors.MSTeams.v1 | Microsoft Teams Outbound Connector | 0.49730828404426575 |
| 23 | SEND EMAIL | bpmn:Task | 3 | io.camunda.connectors.MSFT.O365.Mail.v1 | Microsoft Office 365 Mail Connector | 0.4485277831554413 |
| 23 | SEND EMAIL | bpmn:Task | 4 | io.camunda.connectors.AWSSQS.v1 | Amazon SQS Outbound Connector | 0.4445931017398834 |
| 23 | SEND EMAIL | bpmn:Task | 5 | io.camunda.connectors.WhatsApp.v1 | WhatsApp Business Outbound Connector | 0.434861958026886 |
| 24 | Call external API to get data | bpmn:Task | 1 | io.camunda.connectors.Salesforce.v1 | Salesforce Outbound Connector | 0.6422710418701172 |
| 24 | Call external API to get data | bpmn:Task | 2 | io.camunda.connectors.HttpJson.v2 | REST Outbound Connector | 0.6292446255683899 |
| 24 | Call external API to get data | bpmn:Task | 3 | io.camunda.connectors.CamundaOperate.v1 | Camunda Operate Outbound connector | 0.5950732231140137 |
| 24 | Call external API to get data | bpmn:Task | 4 | io.camunda.connectors.Twilio.v1 | Twilio Outbound Connector | 0.5066078901290894 |
| 24 | Call external API to get data | bpmn:Task | 5 | io.camunda.connectors.AutomationAnywhere | Automation Anywhere Outbound Connector | 0.4664425253868103 |
| 25 | Call Google Maps API | bpmn:Task | 1 | io.camunda.connectors.GoogleMapsPlatform.v1 | Google Maps Platform Outbound Connector | 0.4975009560585022 |
| 25 | Call Google Maps API | bpmn:Task | 2 | io.camunda.connectors.HttpJson.v2 | REST Outbound Connector | 0.48598748445510864 |
| 25 | Call Google Maps API | bpmn:Task | 3 | io.camunda.connectors.Salesforce.v1 | Salesforce Outbound Connector | 0.47507697343826294 |
| 25 | Call Google Maps API | bpmn:Task | 4 | io.camunda.connectors.CamundaOperate.v1 | Camunda Operate Outbound connector | 0.41267797350883484 |
| 25 | Call Google Maps API | bpmn:Task | 5 | io.camunda.connectors.AWSLAMBDA.v2 | AWS Lambda Outbound Connector | 0.40169262886047363 |
| 26 | Call some API | bpmn:Task | 1 | io.camunda.connectors.HttpJson.v2 | REST Outbound Connector | 0.6126003265380859 |
| 26 | Call some API | bpmn:Task | 2 | io.camunda.connectors.Salesforce.v1 | Salesforce Outbound Connector | 0.6066854596138 |
| 26 | Call some API | bpmn:Task | 3 | io.camunda.connectors.CamundaOperate.v1 | Camunda Operate Outbound connector | 0.5031915903091431 |
| 26 | Call some API | bpmn:Task | 4 | io.camunda.connectors.AWSLAMBDA.v2 | AWS Lambda Outbound Connector | 0.4603714942932129 |
| 26 | Call some API | bpmn:Task | 5 | io.camunda.connectors.AutomationAnywhere | Automation Anywhere Outbound Connector | 0.45467159152030945 |
| 27 | Process payment | bpmn:Task | 1 | io.camunda.connectors.Salesforce.v1 | Salesforce Outbound Connector | 0.39762550592422485 |
| 27 | Process payment | bpmn:Task | 2 | io.camunda.connectors.AutomationAnywhere | Automation Anywhere Outbound Connector | 0.33728569746017456 |
| 27 | Process payment | bpmn:Task | 3 | io.camunda.connectors.AzureOpenAI.outbound.v1 | Azure OpenAI Connector | 0.3361597955226898 |
| 27 | Process payment | bpmn:Task | 4 | io.camunda.connectors.MSFT.O365.Mail.v1 | Microsoft Office 365 Mail Connector | 0.3203372657299042 |
| 27 | Process payment | bpmn:Task | 5 | io.camunda.connectors.Jdbc.v1 | SQL Database Connector | 0.3018324673175812 |
| 28 | User registration | bpmn:StartEvent | 1 | io.camunda.connectors.inbound.RabbitMQ.MessageStart.v1 | RabbitMQ Message Start Event Connector | 0.2553573250770569 |
| 28 | User registration | bpmn:StartEvent | 2 | io.camunda.connectors.AWSSQS.StartEvent.v1 | Amazon SQS Start Event Connector | 0.25472983717918396 |
| 28 | User registration | bpmn:StartEvent | 3 | io.camunda.connectors.inbound.RabbitMQ.StartEvent.v1 | RabbitMQ Start Event Connector | 0.2501620054244995 |
| 28 | User registration | bpmn:StartEvent | 4 | io.camunda.connectors.webhook.WebhookConnectorStartMessage.v1 | Webhook Message Start Event Connector | 0.2491978108882904 |
| 28 | User registration | bpmn:StartEvent | 5 | io.camunda.connectors.AWSSQS.startmessage.v1 | Amazon SQS Message Start Event Connector | 0.24719476699829102 |
| 29 | Timer-based reminder | bpmn:IntermediateCatchEvent | 1 | io.camunda.connectors.http.Polling | HTTP Polling Intermediate Catch Event Connector | 0.3098393678665161 |
| 29 | Timer-based reminder | bpmn:IntermediateCatchEvent | 2 | io.camunda.connectors.AWSSQS.intermediate.v1 | Amazon SQS Intermediate Message Catch Event connector | 0.2801808714866638 |
| 29 | Timer-based reminder | bpmn:IntermediateCatchEvent | 3 | io.camunda.connectors.Twilio.Webhook.Intermediate.v1 | Twilio Intermediate Catch Event Connector | 0.2583617568016052 |
| 29 | Timer-based reminder | bpmn:IntermediateCatchEvent | 4 | io.camunda.connectors.inbound.Slack.IntermediateCatchEvent.v1 | Slack Webhook Intermediate Catch Event Connector | 0.24411462247371674 |
| 29 | Timer-based reminder | bpmn:IntermediateCatchEvent | 5 | io.camunda.connectors.webhook.WebhookConnectorIntermediate.v1 | Webhook Intermediate Event Connector | 0.2414102852344513 |
| 30 | Log system event | bpmn:IntermediateThrowEvent | 1 | io.camunda.connectors.AWSSQS.intermediate.v1 | Amazon SQS Intermediate Message Catch Event connector | 0.38605499267578125 |
| 30 | Log system event | bpmn:IntermediateThrowEvent | 2 | io.camunda.connectors.inbound.RabbitMQ.Intermediate.v1 | RabbitMQ Intermediate Catch Event Connector | 0.3842337727546692 |
| 30 | Log system event | bpmn:IntermediateThrowEvent | 3 | io.camunda.connectors.inbound.KafkaIntermediate.v1 | Kafka Intermediate Catch Event Connector | 0.36993494629859924 |
| 30 | Log system event | bpmn:IntermediateThrowEvent | 4 | io.camunda.connectors.AWSEventBridge.intermediate.v1 | Amazon EventBridge Intermediate Catch Event Connector | 0.3678283989429474 |
| 30 | Log system event | bpmn:IntermediateThrowEvent | 5 | io.camunda.connectors.http.Polling | HTTP Polling Intermediate Catch Event Connector | 0.3554941415786743 |
| 31 | Compensate transaction | bpmn:BoundaryEvent | 1 | io.camunda.connectors.webhook.WebhookConnectorBoundary.v1 | Webhook Boundary Event Connector | 0.31111952662467957 |
| 31 | Compensate transaction | bpmn:BoundaryEvent | 2 | io.camunda.connectors.AWSSQS.boundary.v1 | Amazon SQS Boundary Event Connector | 0.30166324973106384 |
| 31 | Compensate transaction | bpmn:BoundaryEvent | 3 | io.camunda.connectors.inbound.RabbitMQ.Boundary.v1 | RabbitMQ Boundary Event Connector | 0.28780820965766907 |
| 31 | Compensate transaction | bpmn:BoundaryEvent | 4 | io.camunda.connectors.http.Polling.Boundary | HTTP Polling Boundary Catch Event Connector | 0.28245794773101807 |
| 31 | Compensate transaction | bpmn:BoundaryEvent | 5 | io.camunda.connectors.inbound.KafkaBoundary.v1 | Kafka Boundary Event Connector | 0.2665276825428009 |
| 32 | Send email with attachment | bpmn:Task | 1 | io.camunda.connectors.SendGrid.v2 | SendGrid Outbound Connector | 0.5608822107315063 |
| 32 | Send email with attachment | bpmn:Task | 2 | io.camunda.connectors.MSFT.O365.Mail.v1 | Microsoft Office 365 Mail Connector | 0.4283294677734375 |
| 32 | Send email with attachment | bpmn:Task | 3 | io.camunda.connectors.MSTeams.v1 | Microsoft Teams Outbound Connector | 0.3904428482055664 |
| 32 | Send email with attachment | bpmn:Task | 4 | io.camunda.connectors.GoogleDrive.v1 | Google Drive Outbound Connector | 0.362640917301178 |
| 32 | Send email with attachment | bpmn:Task | 5 | io.camunda.connectors.Twilio.v1 | Twilio Outbound Connector | 0.35480862855911255 |
| 33 | Wait for customer approval | bpmn:IntermediateCatchEvent | 1 | io.camunda.connectors.AWSSQS.intermediate.v1 | Amazon SQS Intermediate Message Catch Event connector | 0.38343173265457153 |
| 33 | Wait for customer approval | bpmn:IntermediateCatchEvent | 2 | io.camunda.connectors.inbound.RabbitMQ.Intermediate.v1 | RabbitMQ Intermediate Catch Event Connector | 0.31932246685028076 |
| 33 | Wait for customer approval | bpmn:IntermediateCatchEvent | 3 | io.camunda.connectors.inbound.KafkaIntermediate.v1 | Kafka Intermediate Catch Event Connector | 0.3016860783100128 |
| 33 | Wait for customer approval | bpmn:IntermediateCatchEvent | 4 | io.camunda.connectors.Twilio.Webhook.Intermediate.v1 | Twilio Intermediate Catch Event Connector | 0.2997937500476837 |
| 33 | Wait for customer approval | bpmn:IntermediateCatchEvent | 5 | io.camunda.connectors.webhook.WebhookConnectorIntermediate.v1 | Webhook Intermediate Event Connector | 0.2855038046836853 |
| 34 | Error handling mechanism | bpmn:BoundaryEvent | 1 | io.camunda.connectors.http.Polling.Boundary | HTTP Polling Boundary Catch Event Connector | 0.3399356007575989 |
| 34 | Error handling mechanism | bpmn:BoundaryEvent | 2 | io.camunda.connectors.AWSSQS.boundary.v1 | Amazon SQS Boundary Event Connector | 0.3353523313999176 |
| 34 | Error handling mechanism | bpmn:BoundaryEvent | 3 | io.camunda.connectors.inbound.KafkaBoundary.v1 | Kafka Boundary Event Connector | 0.33194607496261597 |
| 34 | Error handling mechanism | bpmn:BoundaryEvent | 4 | io.camunda.connectors.webhook.WebhookConnectorBoundary.v1 | Webhook Boundary Event Connector | 0.32392939925193787 |
| 34 | Error handling mechanism | bpmn:BoundaryEvent | 5 | io.camunda.connectors.inbound.RabbitMQ.Boundary.v1 | RabbitMQ Boundary Event Connector | 0.3112605810165405 |
| 35 | Compensate failed transaction | bpmn:BoundaryEvent | 1 | io.camunda.connectors.AWSSQS.boundary.v1 | Amazon SQS Boundary Event Connector | 0.2969348430633545 |
| 35 | Compensate failed transaction | bpmn:BoundaryEvent | 2 | io.camunda.connectors.webhook.WebhookConnectorBoundary.v1 | Webhook Boundary Event Connector | 0.2846190631389618 |
| 35 | Compensate failed transaction | bpmn:BoundaryEvent | 3 | io.camunda.connectors.inbound.RabbitMQ.Boundary.v1 | RabbitMQ Boundary Event Connector | 0.2731054425239563 |
| 35 | Compensate failed transaction | bpmn:BoundaryEvent | 4 | io.camunda.connectors.http.Polling.Boundary | HTTP Polling Boundary Catch Event Connector | 0.2622634172439575 |
| 35 | Compensate failed transaction | bpmn:BoundaryEvent | 5 | io.camunda.connectors.inbound.KafkaBoundary.v1 | Kafka Boundary Event Connector | 0.2604234218597412 |
| 36 | Log important event | bpmn:IntermediateThrowEvent | 1 | io.camunda.connectors.inbound.RabbitMQ.Intermediate.v1 | RabbitMQ Intermediate Catch Event Connector | 0.34125369787216187 |
| 36 | Log important event | bpmn:IntermediateThrowEvent | 2 | io.camunda.connectors.webhook.GithubWebhookConnectorIntermediate.v1 | GitHub Webhook Intermediate Catch Event Connector | 0.329281210899353 |
| 36 | Log important event | bpmn:IntermediateThrowEvent | 3 | io.camunda.connectors.inbound.KafkaIntermediate.v1 | Kafka Intermediate Catch Event Connector | 0.3275563716888428 |
| 36 | Log important event | bpmn:IntermediateThrowEvent | 4 | io.camunda.connectors.AWSSQS.intermediate.v1 | Amazon SQS Intermediate Message Catch Event connector | 0.32584184408187866 |
| 36 | Log important event | bpmn:IntermediateThrowEvent | 5 | io.camunda.connectors.inbound.Slack.IntermediateCatchEvent.v1 | Slack Webhook Intermediate Catch Event Connector | 0.3198736906051636 |
| 37 | Start process on system boot | bpmn:StartEvent | 1 | io.camunda.connectors.inbound.KafkaMessageStart.v1 | Kafka Message Start Event Connector | 0.3940008878707886 |
| 37 | Start process on system boot | bpmn:StartEvent | 2 | io.camunda.connectors.TwilioWebhook.v1 | Twilio Start Event Connector | 0.3841473460197449 |
| 37 | Start process on system boot | bpmn:StartEvent | 3 | io.camunda.connectors.inbound.KAFKA.v1 | Kafka Start Event Connector | 0.37909960746765137 |
| 37 | Start process on system boot | bpmn:StartEvent | 4 | io.camunda.connectors.TwilioWebhookMessageStart.v1 | Twilio Message Start Event Connector | 0.37865766882896423 |
| 37 | Start process on system boot | bpmn:StartEvent | 5 | io.camunda.connectors.webhook.WebhookConnector.v1 | Webhook Start Event Connector | 0.3731493353843689 |
| 38 | User logs in successfully | bpmn:StartEvent | 1 | io.camunda.connectors.AWSSQS.StartEvent.v1 | Amazon SQS Start Event Connector | 0.27005138993263245 |
| 38 | User logs in successfully | bpmn:StartEvent | 2 | io.camunda.connectors.webhook.WebhookConnectorStartMessage.v1 | Webhook Message Start Event Connector | 0.2540048658847809 |
| 38 | User logs in successfully | bpmn:StartEvent | 3 | io.camunda.connectors.TwilioWebhook.v1 | Twilio Start Event Connector | 0.24749799072742462 |
| 38 | User logs in successfully | bpmn:StartEvent | 4 | io.camunda.connectors.AWSSQS.startmessage.v1 | Amazon SQS Message Start Event Connector | 0.24593797326087952 |
| 38 | User logs in successfully | bpmn:StartEvent | 5 | io.camunda.connectors.webhook.WebhookConnector.v1 | Webhook Start Event Connector | 0.24363696575164795 |
| 39 | Timer event after 24 hours | bpmn:IntermediateCatchEvent | 1 | io.camunda.connectors.http.Polling | HTTP Polling Intermediate Catch Event Connector | 0.33025574684143066 |
| 39 | Timer event after 24 hours | bpmn:IntermediateCatchEvent | 2 | io.camunda.connectors.inbound.Slack.IntermediateCatchEvent.v1 | Slack Webhook Intermediate Catch Event Connector | 0.2781282961368561 |
| 39 | Timer event after 24 hours | bpmn:IntermediateCatchEvent | 3 | io.camunda.connectors.webhook.WebhookConnectorIntermediate.v1 | Webhook Intermediate Event Connector | 0.27362772822380066 |
| 39 | Timer event after 24 hours | bpmn:IntermediateCatchEvent | 4 | io.camunda.connectors.webhook.GithubWebhookConnectorIntermediate.v1 | GitHub Webhook Intermediate Catch Event Connector | 0.2631267309188843 |
| 39 | Timer event after 24 hours | bpmn:IntermediateCatchEvent | 5 | io.camunda.connectors.AWSSQS.intermediate.v1 | Amazon SQS Intermediate Message Catch Event connector | 0.26211512088775635 |
| 40 | Escalate to manager | bpmn:IntermediateThrowEvent | 1 | io.camunda.connectors.AWSSQS.intermediate.v1 | Amazon SQS Intermediate Message Catch Event connector | 0.2939310669898987 |
| 40 | Escalate to manager | bpmn:IntermediateThrowEvent | 2 | io.camunda.connectors.inbound.RabbitMQ.Intermediate.v1 | RabbitMQ Intermediate Catch Event Connector | 0.27319562435150146 |
| 40 | Escalate to manager | bpmn:IntermediateThrowEvent | 3 | io.camunda.connectors.webhook.WebhookConnectorIntermediate.v1 | Webhook Intermediate Event Connector | 0.2450009286403656 |
| 40 | Escalate to manager | bpmn:IntermediateThrowEvent | 4 | io.camunda.connectors.inbound.KafkaIntermediate.v1 | Kafka Intermediate Catch Event Connector | 0.23630337417125702 |
| 40 | Escalate to manager | bpmn:IntermediateThrowEvent | 5 | io.camunda.connectors.inbound.Slack.IntermediateCatchEvent.v1 | Slack Webhook Intermediate Catch Event Connector | 0.23433026671409607 |
| 41 | Validate user input, store results in DB, then trigger notification email | bpmn:Task | 1 | io.camunda.connectors.AutomationAnywhere | Automation Anywhere Outbound Connector | 0.37595421075820923 |
| 41 | Validate user input, store results in DB, then trigger notification email | bpmn:Task | 2 | io.camunda.connectors.Jdbc.v1 | SQL Database Connector | 0.36569392681121826 |
| 41 | Validate user input, store results in DB, then trigger notification email | bpmn:Task | 3 | io.camunda.connectors.UIPath.v1 | UiPath Outbound Connector | 0.36525294184684753 |
| 41 | Validate user input, store results in DB, then trigger notification email | bpmn:Task | 4 | io.camunda.connectors.MSFT.O365.Mail.v1 | Microsoft Office 365 Mail Connector | 0.35514166951179504 |
| 41 | Validate user input, store results in DB, then trigger notification email | bpmn:Task | 5 | io.camunda.connectors.BluePrism.v1 | Blue Prism Outbound Connector | 0.31689688563346863 |
| 42 | Initiate Payment Process | bpmn:StartEvent | 1 | io.camunda.connectors.webhook.WebhookConnector.v1 | Webhook Start Event Connector | 0.38880249857902527 |
| 42 | Initiate Payment Process | bpmn:StartEvent | 2 | io.camunda.connectors.webhook.WebhookConnectorStartMessage.v1 | Webhook Message Start Event Connector | 0.37959277629852295 |
| 42 | Initiate Payment Process | bpmn:StartEvent | 3 | io.camunda.connectors.AWSSQS.StartEvent.v1 | Amazon SQS Start Event Connector | 0.3793686032295227 |
| 42 | Initiate Payment Process | bpmn:StartEvent | 4 | io.camunda.connectors.TwilioWebhook.v1 | Twilio Start Event Connector | 0.379093199968338 |
| 42 | Initiate Payment Process | bpmn:StartEvent | 5 | io.camunda.connectors.inbound.RabbitMQ.MessageStart.v1 | RabbitMQ Message Start Event Connector | 0.373930960893631 |
| 43 | Data Synchronization Timeout | bpmn:IntermediateThrowEvent | 1 | io.camunda.connectors.http.Polling | HTTP Polling Intermediate Catch Event Connector | 0.2826578915119171 |
| 43 | Data Synchronization Timeout | bpmn:IntermediateThrowEvent | 2 | io.camunda.connectors.AWSSQS.intermediate.v1 | Amazon SQS Intermediate Message Catch Event connector | 0.26997604966163635 |
| 43 | Data Synchronization Timeout | bpmn:IntermediateThrowEvent | 3 | io.camunda.connectors.inbound.Slack.IntermediateCatchEvent.v1 | Slack Webhook Intermediate Catch Event Connector | 0.26819831132888794 |
| 43 | Data Synchronization Timeout | bpmn:IntermediateThrowEvent | 4 | io.camunda.connectors.inbound.AWSSNS.IntermediateCatchEvent.v1 | SNS HTTPS Intermediate Catch Event Connector | 0.2652805745601654 |
| 43 | Data Synchronization Timeout | bpmn:IntermediateThrowEvent | 5 | io.camunda.connectors.webhook.GithubWebhookConnectorIntermediate.v1 | GitHub Webhook Intermediate Catch Event Connector | 0.2625950872898102 |
| 44 | Boundary Timer for SLA breach | bpmn:BoundaryEvent | 1 | io.camunda.connectors.http.Polling.Boundary | HTTP Polling Boundary Catch Event Connector | 0.45934104919433594 |
| 44 | Boundary Timer for SLA breach | bpmn:BoundaryEvent | 2 | io.camunda.connectors.AWSSQS.boundary.v1 | Amazon SQS Boundary Event Connector | 0.4577183127403259 |
| 44 | Boundary Timer for SLA breach | bpmn:BoundaryEvent | 3 | io.camunda.connectors.inbound.KafkaBoundary.v1 | Kafka Boundary Event Connector | 0.44635626673698425 |
| 44 | Boundary Timer for SLA breach | bpmn:BoundaryEvent | 4 | io.camunda.connectors.webhook.WebhookConnectorBoundary.v1 | Webhook Boundary Event Connector | 0.41800981760025024 |
| 44 | Boundary Timer for SLA breach | bpmn:BoundaryEvent | 5 | io.camunda.connectors.inbound.Slack.BoundaryEvent.v1 | Slack Webhook Boundary Event Connector | 0.4080730080604553 |
| 45 | Review & Approve Contract Documents | bpmn:Task | 1 | io.camunda.connectors.AutomationAnywhere | Automation Anywhere Outbound Connector | 0.393147736787796 |
| 45 | Review & Approve Contract Documents | bpmn:Task | 2 | io.camunda.connectors.GitLab.v1 | GitLab Outbound Connector | 0.3629009425640106 |
| 45 | Review & Approve Contract Documents | bpmn:Task | 3 | io.camunda.connectors.EasyPost.v1 | Easy Post Outbound Connector | 0.3594457507133484 |
| 45 | Review & Approve Contract Documents | bpmn:Task | 4 | io.camunda.connectors.BluePrism.v1 | Blue Prism Outbound Connector | 0.3530234098434448 |
| 45 | Review & Approve Contract Documents | bpmn:Task | 5 | io.camunda.connectors.UIPath.v1 | UiPath Outbound Connector | 0.34165576100349426 |
| 46 | Wait for 2 hours | bpmn:IntermediateCatchEvent | 1 | io.camunda.connectors.AWSSQS.intermediate.v1 | Amazon SQS Intermediate Message Catch Event connector | 0.40057897567749023 |
| 46 | Wait for 2 hours | bpmn:IntermediateCatchEvent | 2 | io.camunda.connectors.Twilio.Webhook.Intermediate.v1 | Twilio Intermediate Catch Event Connector | 0.36693108081817627 |
| 46 | Wait for 2 hours | bpmn:IntermediateCatchEvent | 3 | io.camunda.connectors.http.Polling | HTTP Polling Intermediate Catch Event Connector | 0.36350443959236145 |
| 46 | Wait for 2 hours | bpmn:IntermediateCatchEvent | 4 | io.camunda.connectors.webhook.WebhookConnectorIntermediate.v1 | Webhook Intermediate Event Connector | 0.3586248457431793 |
| 46 | Wait for 2 hours | bpmn:IntermediateCatchEvent | 5 | io.camunda.connectors.webhook.GithubWebhookConnectorIntermediate.v1 | GitHub Webhook Intermediate Catch Event Connector | 0.3437371253967285 |
| 47 | Callback to External System with JSON Payload | bpmn:Task | 1 | io.camunda.connectors.HttpJson.v2 | REST Outbound Connector | 0.5818662047386169 |
| 47 | Callback to External System with JSON Payload | bpmn:Task | 2 | io.camunda.connectors.Salesforce.v1 | Salesforce Outbound Connector | 0.5782400965690613 |
| 47 | Callback to External System with JSON Payload | bpmn:Task | 3 | io.camunda.connectors.CamundaOperate.v1 | Camunda Operate Outbound connector | 0.47505462169647217 |
| 47 | Callback to External System with JSON Payload | bpmn:Task | 4 | io.camunda.connectors.AWSLAMBDA.v2 | AWS Lambda Outbound Connector | 0.46025586128234863 |
| 47 | Callback to External System with JSON Payload | bpmn:Task | 5 | io.camunda.connectors.KAFKA.v1 | Kafka Outbound Connector | 0.4441756010055542 |
| 48 | N/A | bpmn:Task | 1 | io.camunda.connectors.GoogleSheets.v1 | Google Sheets Outbound Connector | 0.23430316150188446 |
| 48 | N/A | bpmn:Task | 2 | io.camunda.connectors.AutomationAnywhere | Automation Anywhere Outbound Connector | 0.23052863776683807 |
| 48 | N/A | bpmn:Task | 3 | io.camunda.connectors.Asana.v1 | Asana Outbound Connector | 0.22935396432876587 |
| 48 | N/A | bpmn:Task | 4 | io.camunda.connectors.BluePrism.v1 | Blue Prism Outbound Connector | 0.2262793481349945 |
| 48 | N/A | bpmn:Task | 5 | io.camunda.connectors.GraphQL.v1 | GraphQL Outbound Connector | 0.22128476202487946 |
| 49 |     | bpmn:StartEvent | 1 | io.camunda.connectors.webhook.GithubWebhookConnectorMessageStart.v1 | GitHub Webhook Message Start Event Connector | 0.290319561958313 |
| 49 |     | bpmn:StartEvent | 2 | io.camunda.connectors.inbound.RabbitMQ.MessageStart.v1 | RabbitMQ Message Start Event Connector | 0.2726278305053711 |
| 49 |     | bpmn:StartEvent | 3 | io.camunda.connectors.webhook.GithubWebhookConnector.v1 | GitHub Webhook Start Event Connector | 0.27138346433639526 |
| 49 |     | bpmn:StartEvent | 4 | io.camunda.connectors.inbound.AWSSNS.MessageStartEvent.v1 | SNS HTTPS Message Start Event Connector Subscription | 0.2657302916049957 |
| 49 |     | bpmn:StartEvent | 5 | io.camunda.connectors.inbound.RabbitMQ.StartEvent.v1 | RabbitMQ Start Event Connector | 0.2605224549770355 |
| 50 | boundary_event_12345!?$ | bpmn:BoundaryEvent | 1 | io.camunda.connectors.inbound.RabbitMQ.Boundary.v1 | RabbitMQ Boundary Event Connector | 0.5071941614151001 |
| 50 | boundary_event_12345!?$ | bpmn:BoundaryEvent | 2 | io.camunda.connectors.AWSSQS.boundary.v1 | Amazon SQS Boundary Event Connector | 0.49111515283584595 |
| 50 | boundary_event_12345!?$ | bpmn:BoundaryEvent | 3 | io.camunda.connectors.webhook.WebhookConnectorBoundary.v1 | Webhook Boundary Event Connector | 0.46672147512435913 |
| 50 | boundary_event_12345!?$ | bpmn:BoundaryEvent | 4 | io.camunda.connectors.AWSEventBridge.boundary.v1 | Amazon EventBridge Boundary Event Connector | 0.45384228229522705 |
| 50 | boundary_event_12345!?$ | bpmn:BoundaryEvent | 5 | io.camunda.connectors.inbound.KafkaBoundary.v1 | Kafka Boundary Event Connector | 0.44940707087516785 |
| 51 | Fire-and-Forget Notification | bpmn:IntermediateThrowEvent | 1 | io.camunda.connectors.AWSSQS.intermediate.v1 | Amazon SQS Intermediate Message Catch Event connector | 0.4115372598171234 |
| 51 | Fire-and-Forget Notification | bpmn:IntermediateThrowEvent | 2 | io.camunda.connectors.inbound.Slack.IntermediateCatchEvent.v1 | Slack Webhook Intermediate Catch Event Connector | 0.35506966710090637 |
| 51 | Fire-and-Forget Notification | bpmn:IntermediateThrowEvent | 3 | io.camunda.connectors.inbound.KafkaIntermediate.v1 | Kafka Intermediate Catch Event Connector | 0.3376561403274536 |
| 51 | Fire-and-Forget Notification | bpmn:IntermediateThrowEvent | 4 | io.camunda.connectors.inbound.RabbitMQ.Intermediate.v1 | RabbitMQ Intermediate Catch Event Connector | 0.3362691402435303 |
| 51 | Fire-and-Forget Notification | bpmn:IntermediateThrowEvent | 5 | io.camunda.connectors.Twilio.Webhook.Intermediate.v1 | Twilio Intermediate Catch Event Connector | 0.3325425982475281 |
| 52 | Escalate to Human Operator if No Response in 24h | bpmn:IntermediateThrowEvent | 1 | io.camunda.connectors.AWSSQS.intermediate.v1 | Amazon SQS Intermediate Message Catch Event connector | 0.33576685190200806 |
| 52 | Escalate to Human Operator if No Response in 24h | bpmn:IntermediateThrowEvent | 2 | io.camunda.connectors.http.Polling | HTTP Polling Intermediate Catch Event Connector | 0.314741313457489 |
| 52 | Escalate to Human Operator if No Response in 24h | bpmn:IntermediateThrowEvent | 3 | io.camunda.connectors.webhook.WebhookConnectorIntermediate.v1 | Webhook Intermediate Event Connector | 0.2933690547943115 |
| 52 | Escalate to Human Operator if No Response in 24h | bpmn:IntermediateThrowEvent | 4 | io.camunda.connectors.inbound.RabbitMQ.Intermediate.v1 | RabbitMQ Intermediate Catch Event Connector | 0.28877317905426025 |
| 52 | Escalate to Human Operator if No Response in 24h | bpmn:IntermediateThrowEvent | 5 | io.camunda.connectors.Twilio.Webhook.Intermediate.v1 | Twilio Intermediate Catch Event Connector | 0.28483617305755615 |
| 53 | Check XYZ System’s Health and Retry Connection | bpmn:Task | 1 | io.camunda.connectors.CamundaOperate.v1 | Camunda Operate Outbound connector | 0.40437909960746765 |
| 53 | Check XYZ System’s Health and Retry Connection | bpmn:Task | 2 | io.camunda.connectors.BluePrism.v1 | Blue Prism Outbound Connector | 0.40206658840179443 |
| 53 | Check XYZ System’s Health and Retry Connection | bpmn:Task | 3 | io.camunda.connectors.GoogleSheets.v1 | Google Sheets Outbound Connector | 0.3788553774356842 |
| 53 | Check XYZ System’s Health and Retry Connection | bpmn:Task | 4 | io.camunda.connectors.KAFKA.v1 | Kafka Outbound Connector | 0.37708646059036255 |
| 53 | Check XYZ System’s Health and Retry Connection | bpmn:Task | 5 | io.camunda.connectors.UIPath.v1 | UiPath Outbound Connector | 0.37488454580307007 |
| 54 | 非常に長い名前のタスク - 国際化テスト (Internationalization Test) | bpmn:Task | 1 | io.camunda.connectors.GoogleMapsPlatform.v1 | Google Maps Platform Outbound Connector | 0.27135351300239563 |
| 54 | 非常に長い名前のタスク - 国際化テスト (Internationalization Test) | bpmn:Task | 2 | io.camunda.connectors.HuggingFace.v1 | Hugging Face Outbound Connector | 0.21506761014461517 |
| 54 | 非常に長い名前のタスク - 国際化テスト (Internationalization Test) | bpmn:Task | 3 | io.camunda.connectors.WhatsApp.v1 | WhatsApp Business Outbound Connector | 0.20065216720104218 |
| 54 | 非常に長い名前のタスク - 国際化テスト (Internationalization Test) | bpmn:Task | 4 | io.camunda.connectors.GraphQL.v1 | GraphQL Outbound Connector | 0.2003331184387207 |
| 54 | 非常に長い名前のタスク - 国際化テスト (Internationalization Test) | bpmn:Task | 5 | io.camunda.connectors.OpenAI.v1 | OpenAI Outbound Connector | 0.2000426948070526 |
| 55 | Multiple Approvals from Different Departments Required | bpmn:IntermediateCatchEvent | 1 | io.camunda.connectors.inbound.KafkaIntermediate.v1 | Kafka Intermediate Catch Event Connector | 0.21048733592033386 |
| 55 | Multiple Approvals from Different Departments Required | bpmn:IntermediateCatchEvent | 2 | io.camunda.connectors.AWSSQS.intermediate.v1 | Amazon SQS Intermediate Message Catch Event connector | 0.20620855689048767 |
| 55 | Multiple Approvals from Different Departments Required | bpmn:IntermediateCatchEvent | 3 | io.camunda.connectors.webhook.WebhookConnectorIntermediate.v1 | Webhook Intermediate Event Connector | 0.20054741203784943 |
| 55 | Multiple Approvals from Different Departments Required | bpmn:IntermediateCatchEvent | 4 | io.camunda.connectors.webhook.GithubWebhookConnectorIntermediate.v1 | GitHub Webhook Intermediate Catch Event Connector | 0.187667116522789 |
| 55 | Multiple Approvals from Different Departments Required | bpmn:IntermediateCatchEvent | 5 | io.camunda.connectors.inbound.AWSSNS.IntermediateCatchEvent.v1 | SNS HTTPS Intermediate Catch Event Connector | 0.18716350197792053 |
| 56 | -------------------- | bpmn:Task | 1 | io.camunda.connectors.BluePrism.v1 | Blue Prism Outbound Connector | 0.4220983386039734 |
| 56 | -------------------- | bpmn:Task | 2 | io.camunda.connectors.AutomationAnywhere | Automation Anywhere Outbound Connector | 0.40252405405044556 |
| 56 | -------------------- | bpmn:Task | 3 | io.camunda.connectors.UIPath.v1 | UiPath Outbound Connector | 0.3695901334285736 |
| 56 | -------------------- | bpmn:Task | 4 | io.camunda.connectors.MSTeams.v1 | Microsoft Teams Outbound Connector | 0.35902684926986694 |
| 56 | -------------------- | bpmn:Task | 5 | io.camunda.connectors.HuggingFace.v1 | Hugging Face Outbound Connector | 0.34824371337890625 |
| 57 | Generate Monthly Financial Reports & Send PDF to CFO | bpmn:Task | 1 | io.camunda.connectors.GoogleSheets.v1 | Google Sheets Outbound Connector | 0.34389573335647583 |
| 57 | Generate Monthly Financial Reports & Send PDF to CFO | bpmn:Task | 2 | io.camunda.connectors.EasyPost.v1 | Easy Post Outbound Connector | 0.32487571239471436 |
| 57 | Generate Monthly Financial Reports & Send PDF to CFO | bpmn:Task | 3 | io.camunda.connectors.GoogleDrive.v1 | Google Drive Outbound Connector | 0.31594473123550415 |
| 57 | Generate Monthly Financial Reports & Send PDF to CFO | bpmn:Task | 4 | io.camunda.connectors.GitLab.v1 | GitLab Outbound Connector | 0.31194868683815 |
| 57 | Generate Monthly Financial Reports & Send PDF to CFO | bpmn:Task | 5 | io.camunda.connectors.AutomationAnywhere | Automation Anywhere Outbound Connector | 0.30380886793136597 |
| 58 | Initialize Process Flow For Quarterly Audit | bpmn:StartEvent | 1 | io.camunda.connectors.webhook.WebhookConnector.v1 | Webhook Start Event Connector | 0.3615702688694 |
| 58 | Initialize Process Flow For Quarterly Audit | bpmn:StartEvent | 2 | io.camunda.connectors.AWSSQS.StartEvent.v1 | Amazon SQS Start Event Connector | 0.35289084911346436 |
| 58 | Initialize Process Flow For Quarterly Audit | bpmn:StartEvent | 3 | io.camunda.connectors.webhook.WebhookConnectorStartMessage.v1 | Webhook Message Start Event Connector | 0.3462267816066742 |
| 58 | Initialize Process Flow For Quarterly Audit | bpmn:StartEvent | 4 | io.camunda.connectors.AWSSQS.startmessage.v1 | Amazon SQS Message Start Event Connector | 0.33681362867355347 |
| 58 | Initialize Process Flow For Quarterly Audit | bpmn:StartEvent | 5 | io.camunda.connectors.inbound.KafkaMessageStart.v1 | Kafka Message Start Event Connector | 0.321609765291214 |
| 59 | Boundary Error Catching For Database Exception | bpmn:BoundaryEvent | 1 | io.camunda.connectors.http.Polling.Boundary | HTTP Polling Boundary Catch Event Connector | 0.4650164246559143 |
| 59 | Boundary Error Catching For Database Exception | bpmn:BoundaryEvent | 2 | io.camunda.connectors.webhook.WebhookConnectorBoundary.v1 | Webhook Boundary Event Connector | 0.45422279834747314 |
| 59 | Boundary Error Catching For Database Exception | bpmn:BoundaryEvent | 3 | io.camunda.connectors.AWSSQS.boundary.v1 | Amazon SQS Boundary Event Connector | 0.4491780996322632 |
| 59 | Boundary Error Catching For Database Exception | bpmn:BoundaryEvent | 4 | io.camunda.connectors.inbound.RabbitMQ.Boundary.v1 | RabbitMQ Boundary Event Connector | 0.44321349263191223 |
| 59 | Boundary Error Catching For Database Exception | bpmn:BoundaryEvent | 5 | io.camunda.connectors.Twilio.Webhook.Boundary.v1 | Twilio Boundary Event Connector | 0.42849576473236084 |
| 60 | Sensor Trigger - Catch Motion Input | bpmn:IntermediateCatchEvent | 1 | io.camunda.connectors.AWSSQS.intermediate.v1 | Amazon SQS Intermediate Message Catch Event connector | 0.44044196605682373 |
| 60 | Sensor Trigger - Catch Motion Input | bpmn:IntermediateCatchEvent | 2 | io.camunda.connectors.webhook.WebhookConnectorIntermediate.v1 | Webhook Intermediate Event Connector | 0.4262518286705017 |
| 60 | Sensor Trigger - Catch Motion Input | bpmn:IntermediateCatchEvent | 3 | io.camunda.connectors.http.Polling | HTTP Polling Intermediate Catch Event Connector | 0.4157857298851013 |
| 60 | Sensor Trigger - Catch Motion Input | bpmn:IntermediateCatchEvent | 4 | io.camunda.connectors.inbound.KafkaIntermediate.v1 | Kafka Intermediate Catch Event Connector | 0.40974730253219604 |
| 60 | Sensor Trigger - Catch Motion Input | bpmn:IntermediateCatchEvent | 5 | io.camunda.connectors.Twilio.Webhook.Intermediate.v1 | Twilio Intermediate Catch Event Connector | 0.40781334042549133 |
| 61 | Signal Throw to Downstream Systems | bpmn:IntermediateThrowEvent | 1 | io.camunda.connectors.AWSSQS.intermediate.v1 | Amazon SQS Intermediate Message Catch Event connector | 0.4359774887561798 |
| 61 | Signal Throw to Downstream Systems | bpmn:IntermediateThrowEvent | 2 | io.camunda.connectors.AWSEventBridge.intermediate.v1 | Amazon EventBridge Intermediate Catch Event Connector | 0.43321067094802856 |
| 61 | Signal Throw to Downstream Systems | bpmn:IntermediateThrowEvent | 3 | io.camunda.connectors.inbound.RabbitMQ.Intermediate.v1 | RabbitMQ Intermediate Catch Event Connector | 0.4222623109817505 |
| 61 | Signal Throw to Downstream Systems | bpmn:IntermediateThrowEvent | 4 | io.camunda.connectors.inbound.Slack.IntermediateCatchEvent.v1 | Slack Webhook Intermediate Catch Event Connector | 0.40435561537742615 |
| 61 | Signal Throw to Downstream Systems | bpmn:IntermediateThrowEvent | 5 | io.camunda.connectors.Twilio.Webhook.Intermediate.v1 | Twilio Intermediate Catch Event Connector | 0.3923909366130829 |

## Suggestion resuts by OpenAI Function Calling

| Test Case ID | Input Name | Input Type | Suggestion Rank | Suggestion Name | Prompt Tokens | Completion Tokens | Total Tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Send email | bpmn:Task | 1.0 | SendGrid Outbound Connector | 1236 | 18 | 1254 |
| 2 | Send message | bpmn:Task | 1.0 | Slack Outbound Connector | 1236 | 19 | 1255 |
| 3 | Setup project | bpmn:Task | 1.0 | Asana Outbound Connector | 1236 | 18 | 1254 |
| 4 | Call REST API | bpmn:Task | 1.0 | REST Outbound Connector | 1237 | 19 | 1256 |
| 5 | Get distance from Google Maps | bpmn:Task | 1.0 | Google Maps Platform Outbound Connector | 1239 | 20 | 1259 |
| 6 | Send message on Slack | bpmn:Task | 1.0 | Slack Outbound Connector | 1238 | 19 | 1257 |
| 7 | Receive order | bpmn:StartEvent |  | error | 0 | 0 | 0 |
| 8 | Wait for approval | bpmn:IntermediateCatchEvent |  | error | 0 | 0 | 0 |
| 9 | GitHub issue | bpmn:BoundaryEvent | 1.0 | GitHub Webhook Boundary Event Connector | 498 | 24 | 522 |
| 10 | Create Slack channel | bpmn:Task | 1.0 | Slack Outbound Connector | 1237 | 19 | 1256 |
| 11 | Email sender | bpmn:Task | 1.0 | SendGrid Outbound Connector | 1236 | 18 | 1254 |
| 12 | Order received | bpmn:StartEvent |  | error | 0 | 0 | 0 |
| 13 | Approval wait | bpmn:IntermediateCatchEvent |  | error | 0 | 0 | 0 |
| 14 | Notification sender | bpmn:IntermediateThrowEvent |  | error | 0 | 0 | 0 |
| 15 | Send email to customer | bpmn:Task | 1.0 | SendGrid Outbound Connector | 1238 | 18 | 1256 |
| 16 | Make API request | bpmn:Task | 1.0 | REST Outbound Connector | 1237 | 19 | 1256 |
| 17 | Email | bpmn:IntermediateCatchEvent |  | error | 0 | 0 | 0 |
| 18 | Receive Email | bpmn:IntermediateCatchEvent |  | error | 0 | 0 | 0 |
| 19 | Approval | bpmn:Task |  | No suggestions | 1235 | 16 | 1251 |
| 20 | Approval | bpmn:IntermediateCatchEvent |  | error | 0 | 0 | 0 |
| 21 | Send Email Notification | bpmn:Task | 1.0 | SendGrid Outbound Connector | 1237 | 18 | 1255 |
| 22 | Fire John | bpmn:Task | 1.0 | Slack Outbound Connector | 1236 | 19 | 1255 |
| 23 | SEND EMAIL | bpmn:Task | 1.0 | SendGrid Outbound Connector | 1236 | 18 | 1254 |
| 24 | Call external API to get data | bpmn:Task | 1.0 | REST Outbound Connector | 1240 | 19 | 1259 |
| 25 | Call Google Maps API | bpmn:Task | 1.0 | Google Maps Platform Outbound Connector | 1238 | 20 | 1258 |
| 26 | Call some API | bpmn:Task | 1.0 | REST Outbound Connector | 1237 | 80 | 1317 |
| 27 | Process payment | bpmn:Task | 1.0 | Salesforce Outbound Connector | 1236 | 19 | 1255 |
| 28 | User registration | bpmn:StartEvent |  | error | 0 | 0 | 0 |
| 29 | Timer-based reminder | bpmn:IntermediateCatchEvent |  | error | 0 | 0 | 0 |
| 30 | Log system event | bpmn:IntermediateThrowEvent |  | error | 0 | 0 | 0 |
| 31 | Compensate transaction | bpmn:BoundaryEvent |  | No suggestions | 500 | 12 | 512 |
| 32 | Send email with attachment | bpmn:Task | 1.0 | SendGrid Outbound Connector | 1238 | 18 | 1256 |
| 33 | Wait for customer approval | bpmn:IntermediateCatchEvent |  | error | 0 | 0 | 0 |
| 34 | Error handling mechanism | bpmn:BoundaryEvent |  | No suggestions | 499 | 12 | 511 |
| 35 | Compensate failed transaction | bpmn:BoundaryEvent | 1.0 | Webhook Boundary Event Connector | 501 | 23 | 524 |
| 36 | Log important event | bpmn:IntermediateThrowEvent |  | error | 0 | 0 | 0 |
| 37 | Start process on system boot | bpmn:StartEvent |  | error | 0 | 0 | 0 |
| 38 | User logs in successfully | bpmn:StartEvent |  | error | 0 | 0 | 0 |
| 39 | Timer event after 24 hours | bpmn:IntermediateCatchEvent |  | error | 0 | 0 | 0 |
| 40 | Escalate to manager | bpmn:IntermediateThrowEvent |  | error | 0 | 0 | 0 |
| 41 | Validate user input, store results in DB, then trigger notification email | bpmn:Task | 1.0 | REST Outbound Connector | 1247 | 141 | 1388 |
| 41 | Validate user input, store results in DB, then trigger notification email | bpmn:Task | 2.0 | SQL Database Connector | 1247 | 141 | 1388 |
| 41 | Validate user input, store results in DB, then trigger notification email | bpmn:Task | 3.0 | SendGrid Outbound Connector | 1247 | 141 | 1388 |
| 42 | Initiate Payment Process | bpmn:StartEvent |  | error | 0 | 0 | 0 |
| 43 | Data Synchronization Timeout | bpmn:IntermediateThrowEvent |  | error | 0 | 0 | 0 |
| 44 | Boundary Timer for SLA breach | bpmn:BoundaryEvent |  | No suggestions | 502 | 12 | 514 |
| 45 | Review & Approve Contract Documents | bpmn:Task | 1.0 | Google Drive Outbound Connector | 1240 | 19 | 1259 |
| 46 | Wait for 2 hours | bpmn:IntermediateCatchEvent |  | error | 0 | 0 | 0 |
| 47 | Callback to External System with JSON Payload | bpmn:Task | 1.0 | REST Outbound Connector | 1241 | 19 | 1260 |
| 48 | N/A | bpmn:Task |  | No suggestions | 1236 | 37 | 1273 |
| 49 |     | bpmn:StartEvent |  | No suggestions | 0 | 0 | 0 |
| 50 | boundary_event_12345!?$ | bpmn:BoundaryEvent |  | No suggestions | 503 | 12 | 515 |
| 51 | Fire-and-Forget Notification | bpmn:IntermediateThrowEvent |  | error | 0 | 0 | 0 |
| 52 | Escalate to Human Operator if No Response in 24h | bpmn:IntermediateThrowEvent |  | error | 0 | 0 | 0 |
| 53 | Check XYZ System’s Health and Retry Connection | bpmn:Task | 1.0 | REST Outbound Connector | 1242 | 82 | 1324 |
| 54 | 非常に長い名前のタスク - 国際化テスト (Internationalization Test) | bpmn:Task |  | No suggestions | 1258 | 14 | 1272 |
| 55 | Multiple Approvals from Different Departments Required | bpmn:IntermediateCatchEvent |  | error | 0 | 0 | 0 |
| 56 | -------------------- | bpmn:Task |  | No suggestions | 1235 | 35 | 1270 |
| 57 | Generate Monthly Financial Reports & Send PDF to CFO | bpmn:Task | 1.0 | Google Sheets Outbound Connector | 1243 | 108 | 1351 |
| 57 | Generate Monthly Financial Reports & Send PDF to CFO | bpmn:Task | 2.0 | SendGrid Outbound Connector | 1243 | 108 | 1351 |
| 58 | Initialize Process Flow For Quarterly Audit | bpmn:StartEvent |  | error | 0 | 0 | 0 |
| 59 | Boundary Error Catching For Database Exception | bpmn:BoundaryEvent |  | No suggestions | 503 | 12 | 515 |
| 60 | Sensor Trigger - Catch Motion Input | bpmn:IntermediateCatchEvent |  | error | 0 | 0 | 0 |
| 61 | Signal Throw to Downstream Systems | bpmn:IntermediateThrowEvent |  | error | 0 | 0 | 0 |