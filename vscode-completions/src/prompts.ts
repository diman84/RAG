
export const FimPrompts = {
    StableCodePropt: {
        template: "<fim_prefix>{{{prefix}}}<fim_suffix>{{{suffix}}}<fim_middle>"
    },
    OpenAIPrompt: {
        system: `You are coding assistant in JS language and you need to suggest only fill-in-the-middle text formatted as '{{FIM}}'. Your task is to complete with a string to replace this hole applying context-aware indentation, if needed.  All completions MUST be truthful, accurate, well-written and correct. Use ECMASCRIPT 6 whenever possible.

## EXAMPLE QUERY:

function sum_evens(lim) {
  var sum = 0;
  for (var i = 0; i < lim; ++i) {
    {{FIM}}
  }
  return sum;
}

## CORRECT COMPLETION
if (i % 2 === 0) {
      sum += i;
    }

## EXAMPLE QUERY:

function sum_list(lst):
  total = 0
  for x in lst:
  {{FIM}}
  return total

## CORRECT COMPLETION:

 total += x

## EXAMPLE QUERY:

function hypothenuse(a, b) {
  return Math.sqrt({{FIM}}b ** 2);
}

## CORRECT COMPLETION:

a ** 2 +`,
        template: (prefix: string, suffix: string) =>
    {
        return`
Fill the {{FIM}} hole. Answer only with the CORRECT completion, and NOTHING ELSE.
        
${prefix}{{FIM}}${suffix}
`;
        }
    }
}