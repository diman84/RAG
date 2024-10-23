import * as vscode from 'vscode';
import { Range } from 'vscode';
import { getEncoding, encodingForModel } from "js-tiktoken";
import { Indexer } from './indexer';
import { Suggestions } from './suggestions';

function createLoadingIndicator(): vscode.StatusBarItem {
	let li = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 10);
	li.text = "$(loading~spin) GPT";
	li.tooltip = "Generating completions...";
	return li;
}

export function activate(context: vscode.ExtensionContext) {

    const tokenEncoding = encodingForModel("gpt-4o-mini");
    
    const repositoryIndexer = new Indexer();
    repositoryIndexer.indexRepositoryFiles();

    const llm = new Suggestions();
    llm.initialize();
    
    const loadingIndicator = createLoadingIndicator();

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

                const tokenLimit = 1000; // Desired total token length
                let prefix = '';
                let postfix = '';
                let currentLength = tokenEncoding.encode(code.substring(0, position.character)).length;

                let prefixIndex = position.line - 1;
                let postfixIndex = position.line + 1;
                let linePrefix = code.substring(0, position.character);
                let linePostfix = code.substring(position.character);
                prefix = linePrefix;
                postfix = linePostfix;

                // Calculate prefix and postfix simultaneously
                while (currentLength < tokenLimit && (prefixIndex >= 0 || postfixIndex < document.lineCount)) {
                    if (prefixIndex >= 0) {
                        const lineText = document.lineAt(prefixIndex).text;
                        const lineTokens = tokenEncoding.encode(lineText).length;
                        if (currentLength + lineTokens <= tokenLimit) {
                            prefix = lineText + '\n' + prefix;
                            currentLength += lineTokens;
                            prefixIndex--;
                        } else {
                            break;
                        }
                    }

                    if (postfixIndex < document.lineCount) {
                        const lineText = document.lineAt(postfixIndex).text;
                        const lineTokens = tokenEncoding.encode(lineText).length;
                        if (currentLength + lineTokens <= tokenLimit) {
                            postfix += '\n' + lineText;
                            currentLength += lineTokens;
                            postfixIndex++;
                        } else {
                            break;
                        }
                    }

                    if (prefixIndex < 0 && postfixIndex >= document.lineCount) {
                        break;
                    }
                }
                
                loadingIndicator.show();
                let suggestion = await llm.getFimSuggestion(prefix, postfix);
                loadingIndicator.hide();

                result.items.push({
                    insertText: suggestion.replace(/`/gi,'',).trim(),
                    range: new Range(position.line, position.character, position.line, position.character + suggestion.length),
                });

                var relatedDocs = await repositoryIndexer.searchRelatedDocuments(code, 3);

                for (const doc of relatedDocs) {
                    // const suggestion = await llm.getFimSuggestion(doc.pageContent, '');

                    result.items.push({
                        insertText: doc.metadata['source'],
                        range: new Range(position.line, position.character, position.line, position.character + suggestion.length),
                        // add command to open the document
                    });
                }
            }

            return result;
		},
	};
	vscode.languages.registerInlineCompletionItemProvider({ pattern: '**' }, provider);
}