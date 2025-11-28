"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field

from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS

from .models import Message, PromptFunction, PromptVersion
from .prompt_helpers import to_prompt_json
from .snippets import summary_instructions


class ExtractedEntity(BaseModel):
    name: str = Field(..., description='Name of the extracted entity')
    entity_type_id: int = Field(
        description='ID of the classified entity type. '
        'Must be one of the provided entity_type_id integers.',
    )


class ExtractedEntities(BaseModel):
    extracted_entities: list[ExtractedEntity] = Field(..., description='List of extracted entities')


class MissedEntities(BaseModel):
    missed_entities: list[str] = Field(..., description="Names of entities that weren't extracted")


class EntityClassificationTriple(BaseModel):
    uuid: str = Field(description='UUID of the entity')
    name: str = Field(description='Name of the entity')
    entity_type: str | None = Field(
        default=None,
        description='Type of the entity. Must be one of the provided types or None',
    )


class EntityClassification(BaseModel):
    entity_classifications: list[EntityClassificationTriple] = Field(
        ..., description='List of entities classification triples.'
    )


class EntitySummary(BaseModel):
    summary: str = Field(
        ...,
        description=f'Summary containing the important information about the entity. Under {MAX_SUMMARY_CHARS} characters.',
    )


class Prompt(Protocol):
    extract_message: PromptVersion
    extract_json: PromptVersion
    extract_text: PromptVersion
    reflexion: PromptVersion
    classify_nodes: PromptVersion
    extract_attributes: PromptVersion
    extract_summary: PromptVersion


class Versions(TypedDict):
    extract_message: PromptFunction
    extract_json: PromptFunction
    extract_text: PromptFunction
    reflexion: PromptFunction
    classify_nodes: PromptFunction
    extract_attributes: PromptFunction
    extract_summary: PromptFunction


def extract_message(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that extracts entity nodes from conversational messages. 
    Your primary task is to extract and classify the speaker and other significant entities mentioned in the conversation. They are a professional who is using an app to help them remember the details about the people, locations, properties, transactions, tasks, and events that makes them great at their job.
    
    Pay special attention to extracting tasks and to-dos when users mention things they need to do, complete, or remember. These should be classified as Activity entities with appropriate temporal information captured as attributes."""

    user_prompt = f"""
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

<PREVIOUS MESSAGES>
{to_prompt_json([ep for ep in context['previous_episodes']])}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{context['episode_content']}
</CURRENT MESSAGE>

Instructions:

You are given a conversation context and a CURRENT MESSAGE. Your task is to extract **entity nodes** mentioned **explicitly or implicitly** in the CURRENT MESSAGE.
Pronoun references such as he/she/they or this/that/those should be disambiguated to the names of the reference entities. Self references such as "I" or "me" or "myself" should be disambiguated to the speaker. Only extract distinct entities from the CURRENT MESSAGE. Don't extract pronouns like you, me, he/she/they, we/us as entities.

1. **Speaker Extraction**: Always extract the speaker (the part before the colon `:` in each dialogue line) as the first entity node.
   - If the speaker is mentioned again in the message, treat both mentions as a **single entity**.

2. **Entity Identification**:
   - Extract all significant entities, concepts, or actors that are **explicitly or implicitly** mentioned in the CURRENT MESSAGE. Use the PREVIOUS MESSAGES to determine if a reference in the current message (example "she has red hair" the reference being 'she', or "they want a blue house" referencing individuals previously mentioned) is a reference to an entity previously mentioned. To disambiguate those references, consider the last messages as the most recent (example: if first message is "my sister katie" and second message is "my sister julia" and current message is "she has red hair" the reference is to julia, the last-mentioned appropriate entity).
   - **Exclude** entities mentioned only in the PREVIOUS MESSAGES (they are for context only) unless you are certain they are being mentioned in the CURRENT MESSAGE.

3. **Entity Classification**:
   - Use the descriptions in ENTITY TYPES to classify each extracted entity.
   - Assign the appropriate `entity_type_id` for each one.

4. **Exclusions**:
   - Do NOT extract entities representing relationships between other entities.
   - Do NOT extract dates, times, or other temporal information—these will be handled separately.
   - Do NOT extract things that are attributes of entities, such as phone number or email addresses or dollar amounts.
   - **EXCEPTION**: DO extract activity or action entities that represent any actionable items, to-dos, or activities (e.g., "follow up with client", "take out trash", "call mom", "schedule inspection", "go to store").

5. **Formatting**:
   - Be **explicit and unambiguous** in naming entities (e.g., use full names when available).

{context['custom_prompt']}

**UNIFIED ACTIVITY EXTRACTION**:
- **ACTIVITY TRIGGERS**: Extract Activity entities when the message contains any actionable items or to-dos:
  * Action words: schedule, follow up, prepare, review, call, meet, contact, arrange, coordinate, remind, take, go, buy, pick up
  * Any imperative or reminder statements: "remind me to...", "I need to...", "I should..."
- **EXTRACTION STRATEGY**:
  * Activity + mentioned entities = Extract Activity entity + re-extract ALL mentioned people/properties (even if they exist in previous episodes)
  * **CRITICAL**: When extracting Activity entities, always re-extract mentioned people/properties as entities in the same episode to enable automatic edge creation
- **EXAMPLES**:
  * "Follow up with Sarah Chen about mortgage" → Extract: PersonNode(Chris Murray), PersonNode(Sarah Chen), ActivityNode(follow up about mortgage)
  * "Meet with inspector Mike for Johnson property" → Extract: PersonNode(Chris Murray), PersonNode(Mike), PropertyNode(Johnson property), ActivityNode(meet with inspector)
  * "Remind me to tell Lisa about the wedding" → Extract: PersonNode(Chris Murray), PersonNode(Lisa), ActivityNode(tell Lisa about wedding)
  * "Call mom tonight" → Extract: PersonNode(Chris Murray), PersonNode(mom), ActivityNode(call mom)
  * "Remind me to take out trash" → Extract: PersonNode(Chris Murray), ActivityNode(take out trash) - NOTE: Activity connects only to speaker since no other entities mentioned
  * "I need to check the Smith property listing" → Extract: PersonNode(Chris Murray), PropertyNode(Smith property), ActivityNode(check listing)
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_json(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that extracts entity nodes from JSON. 
    Your primary task is to extract and classify relevant entities from JSON files"""

    user_prompt = f"""
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

<SOURCE DESCRIPTION>:
{context['source_description']}
</SOURCE DESCRIPTION>
<JSON>
{context['episode_content']}
</JSON>

{context['custom_prompt']}

Given the above source description and JSON, extract relevant entities from the provided JSON.
For each entity extracted, also determine its entity type based on the provided ENTITY TYPES and their descriptions.
Indicate the classified entity type by providing its entity_type_id.

Guidelines:
1. Extract all entities that the JSON represents. This will often be something like a "name" or "user" field
2. Extract all entities mentioned in all other properties throughout the JSON structure
3. Do NOT extract any properties that contain dates
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_text(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that extracts entity nodes from text. 
    Your primary task is to extract and classify the speaker and other significant entities mentioned in the provided text."""

    user_prompt = f"""
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

<TEXT>
{context['episode_content']}
</TEXT>

Given the above text, extract entities from the TEXT that are explicitly or implicitly mentioned.
For each entity extracted, also determine its entity type based on the provided ENTITY TYPES and their descriptions.
Indicate the classified entity type by providing its entity_type_id.

{context['custom_prompt']}

Guidelines:
1. Extract significant entities, concepts, or actors mentioned in the conversation.
2. Avoid creating nodes for relationships or actions.
3. Avoid creating nodes for temporal information like dates, times or years (these will be added to edges later).
4. Be as explicit as possible in your node names, using full names and avoiding abbreviations.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which entities have not been extracted from the given context"""

    user_prompt = f"""
<PREVIOUS MESSAGES>
{to_prompt_json([ep for ep in context['previous_episodes']])}
</PREVIOUS MESSAGES>
<CURRENT MESSAGE>
{context['episode_content']}
</CURRENT MESSAGE>

<EXTRACTED ENTITIES>
{context['extracted_entities']}
</EXTRACTED ENTITIES>

Given the above previous messages, current message, and list of extracted entities; determine if any entities haven't been
extracted.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def classify_nodes(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that classifies entity nodes given the context from which they were extracted"""

    user_prompt = f"""
    <PREVIOUS MESSAGES>
    {to_prompt_json([ep for ep in context['previous_episodes']])}
    </PREVIOUS MESSAGES>
    <CURRENT MESSAGE>
    {context['episode_content']}
    </CURRENT MESSAGE>

    <EXTRACTED ENTITIES>
    {context['extracted_entities']}
    </EXTRACTED ENTITIES>

    <ENTITY TYPES>
    {context['entity_types']}
    </ENTITY TYPES>

    Given the above conversation, extracted entities, and provided entity types and their descriptions, classify the extracted entities.

    Guidelines:
    1. Each entity must have exactly one type
    2. Only use the provided ENTITY TYPES as types, do not use additional types to classify entities.
    3. If none of the provided entity types accurately classify an extracted node, the type should be set to None
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts entity properties from the provided text.',
        ),
        Message(
            role='user',
            content=f"""

        <MESSAGES>
        {to_prompt_json(context['previous_episodes'])}
        {to_prompt_json(context['episode_content'])}
        </MESSAGES>

        Given the above MESSAGES and the specific ENTITY below, extract attributes ONLY for that ENTITY. 
        - Do NOT extract or infer attributes from other entities or speakers.
        - If there is no clear attribute information about the ENTITY in the MESSAGES, return None or empty values for all attributes.
        - Only set attribute values that are clearly and explicitly related to the ENTITY below.

        Guidelines:
        1. Do not hallucinate entity property values if they cannot be found in the current context.
        2. Only use the provided MESSAGES and ENTITY to set attribute values.
        3. The summary attribute represents a summary of the ENTITY, and should be updated with new information about the Entity from the MESSAGES. Only summarize information specific to this one ENTITY, not all Entities detected in the MESSAGES. Summaries must be no longer than 400 characters.
        4. PRESERVE domain-specific vocabulary: Retain exact role titles, classification categories, technical terms, system names, issue types, and specialized phrases from the source text. These terms are essential for search retrieval. Do not generalize specific terminology.
        
        Example:
        If the ENTITY is "John's dog" and the MESSAGES only mention "John went to the park," then all attributes for an Entity "John's dog" should be None or empty.

        ENTITY TO EXTRACT attributes for:

        <ENTITY>
        {context['node']}
        </ENTITY>
        """,
        ),
    ]


def extract_summary(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts entity summaries from the provided text.',
        ),
        Message(
            role='user',
            content=f"""
        Given the MESSAGES and the ENTITY, update the summary that combines relevant information about the entity
        from the messages and relevant information from the existing summary.

        {summary_instructions}

        <MESSAGES>
        {to_prompt_json(context['previous_episodes'])}
        {to_prompt_json(context['episode_content'])}
        </MESSAGES>

        <ENTITY>
        {context['node']}
        </ENTITY>
        """,
        ),
    ]


versions: Versions = {
    'extract_message': extract_message,
    'extract_json': extract_json,
    'extract_text': extract_text,
    'reflexion': reflexion,
    'extract_summary': extract_summary,
    'classify_nodes': classify_nodes,
    'extract_attributes': extract_attributes,
}
