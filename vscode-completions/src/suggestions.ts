import { ChatOpenAI } from "@langchain/openai";
import { FimPrompts } from "./prompts";

export class Suggestions {
    private chat: ChatOpenAI;

    constructor() {
        this.chat = new ChatOpenAI({
            model: "gpt-4o",
            temperature: 0.5,
            topP: 1,
            maxTokens: 3000,
          });
    }

    async initialize(): Promise<void> {
        await this.chat.invoke([
            {
              role: "system",
              content: FimPrompts.OpenAIPrompt.system,
            }]);
    }

    async getFimSuggestion(prefix: string, suffix: string): Promise<string> {
        const prompt = FimPrompts.OpenAIPrompt.template(prefix, suffix);
        const aiMsg = await this.chat.invoke([
            {
              role: "user",
              content: prompt,
            }]);

        return aiMsg.content.toString();
    }
}