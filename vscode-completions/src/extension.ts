import * as vscode from 'vscode';
import { Range } from 'vscode';

export function activate(context: vscode.ExtensionContext) {

	const provider: vscode.InlineCompletionItemProvider = {
		async provideInlineCompletionItems(document, position, context, token) {
			console.log('provideInlineCompletionItems triggered');
            // match text that is not a comment
            const regexp = /(^(?!\s*\/\/|\/\*|\*\/).+)/g;
			if (position.line <= 0) {
				return;
			}

			const result: vscode.InlineCompletionList = {
				items: [],
			};
            
            const code = document.lineAt(position.line).text;
            const matches = code.match(regexp);
            if (matches) {

                var suggestion = "MY COOL SUGGESTION";

                result.items.push({
                    insertText: suggestion,
                    range: new Range(position.line, position.character, position.line, position.character + suggestion.length),
                    //completeBracketPairs,
                });
            }

            return result;
		},
	};
	vscode.languages.registerInlineCompletionItemProvider({ pattern: '**' }, provider);
}