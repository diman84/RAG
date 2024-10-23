import * as vscode from 'vscode';
import { Range } from 'vscode';
import {
	DocumentFilter,
	LanguageClient,
	LanguageClientOptions,
	ServerOptions,
	TransportKind,
    RequestType,
    ProtocolRequestType
} from 'vscode-languageclient/node';

interface Completion {
	label: string;
}

interface CompletionResponse {
	items: Completion[],
}

let client: LanguageClient;
let ctx: vscode.ExtensionContext;
let loadingIndicator: vscode.StatusBarItem;

function createLoadingIndicator(): vscode.StatusBarItem {
	let li = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 10);
	li.text = "$(loading~spin) LSP";
	li.tooltip = "Generating completions...";
	return li;
}

export async function activate(context: vscode.ExtensionContext) {

    const command = vscode.Uri.joinPath(context.extensionUri, "..", "lsp-server", "dist", "server", "server.exe").fsPath;

    const serverOptions: ServerOptions = {
		run: {
			command,
            transport: TransportKind.stdio,
			options: {
				env: {
					"HF_TOKEN": process.env.HF_TOKEN
				}
			}
		},
		debug: {
			command,
			transport: TransportKind.stdio,
			options: {
				env: {
					"HF_TOKEN": process.env.HF_TOKEN
				}
			}
		}
	};

    const outputChannel = vscode.window.createOutputChannel('LSP VS Code', { log: true });
	const clientOptions: LanguageClientOptions = {
		documentSelector: [{ scheme: "*" }],
		outputChannel,
	};

	client = new LanguageClient(
		'lsp-llm',
		'LSP VS Code',
		serverOptions,
		clientOptions
	);

	loadingIndicator = createLoadingIndicator();

	await client.start();

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

                let params = {
                    position,
                    textDocument: client.code2ProtocolConverter.asTextDocumentIdentifier(document),
                };
                try {

					loadingIndicator.show()
                    const response: CompletionResponse = await client.sendRequest("textDocument/completion", params, token);
					loadingIndicator.hide()

                    for (const completion of response.items) {
                        result.items.push({
                            insertText: completion.label,
                            range: new vscode.Range(position, position),
                        });
                    }
                } catch (e) {
                    const err_msg = (e as Error).message;
                    if (err_msg !== "Canceled") {
                        vscode.window.showErrorMessage(err_msg);
                    }
                }
            }

            return result;
		},
	};
	vscode.languages.registerInlineCompletionItemProvider({ pattern: '**' }, provider);
}