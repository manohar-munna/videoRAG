package com.cctv.videorag.llm

/**
 * One completed exchange, replayed into later prompts so follow-up questions resolve
 * against what was already established ("what colour was it?" after "is there a bus?").
 */
data class ConversationTurn(
    val question: String,
    val answer: String
)
