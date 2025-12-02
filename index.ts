import { anthropic } from '@ai-sdk/anthropic';
import { streamText } from 'ai';
import 'dotenv/config';
import * as readline from 'node:readline/promises';

const terminal = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

type AgentName = 'uiux' | 'frontend' | 'backend';

type MessageKind =
  | 'feature-request'
  | 'uiux-spec'
  | 'frontend-plan'
  | 'backend-plan';

interface BoardMessage {
  id: number;
  author: 'user' | AgentName;
  kind: MessageKind;
  content: string;
  createdAt: Date;
}

interface CapabilityManifest {
  name: string;
  role: string;
  description: string;
  inputs: string[];
  outputs: string[];
  triggerSignatures: string[];
  localPolicy: string;
}

interface Agent {
  name: AgentName;
  manifest: CapabilityManifest;
  /**
   * Inspect the global message board and decide whether to act.
   * Returns the message this agent wants to respond to, or null.
   */
  findTrigger(board: BoardMessage[]): BoardMessage | null;
  /**
   * Run the agent's "act" function with the given trigger message
   * and produce a new message to be posted to the board.
   */
  act(trigger: BoardMessage, board: BoardMessage[]): Promise<BoardMessage>;
}

let nextId = 1;

function createMessage(
  author: BoardMessage['author'],
  kind: MessageKind,
  content: string,
): BoardMessage {
  return {
    id: nextId++,
    author,
    kind,
    content,
    createdAt: new Date(),
  };
}

const uiuxManifest: CapabilityManifest = {
  name: 'UI/UX Design Agent',
  role: 'uiux',
  description:
    'Designs user flows, screens, and interaction patterns based on feature requests. Produces clear, structured UI/UX specs for engineers.',
  inputs: ['feature-request', 'existing UX context on the board'],
  outputs: ['uiux-spec'],
  triggerSignatures: [
    'New feature request without an existing UI/UX spec',
    'User asks for a new flow or change in user experience',
  ],
  localPolicy:
    'Act once per feature request to produce a complete UI/UX spec. Do not re-act unless the feature request changes significantly.',
};

const frontendManifest: CapabilityManifest = {
  name: 'Frontend Engineer Agent',
  role: 'frontend',
  description:
    'Implements UI/UX specifications in a modern frontend stack. Produces implementation plans and component breakdowns.',
  inputs: ['uiux-spec', 'feature-request'],
  outputs: ['frontend-plan'],
  triggerSignatures: [
    'UI/UX specs are posted / ready for review',
    'Frontend work is requested explicitly',
  ],
  localPolicy:
    'Wait until a UI/UX spec exists for the feature. Then translate specs into a concrete implementation plan and surface open questions.',
};

const backendManifest: CapabilityManifest = {
  name: 'Backend Engineer Agent',
  role: 'backend',
  description:
    'Designs and implements APIs, data models, and backend workflows that support the requested feature and the UI/UX design.',
  inputs: ['feature-request', 'uiux-spec'],
  outputs: ['backend-plan'],
  triggerSignatures: [
    'UI/UX specs are posted / ready for review',
    'New data or API requirements implied by the request',
  ],
  localPolicy:
    'Wait until a UI/UX spec exists for the feature. Then design backend contracts, storage, and integration points to support the UI.',
};

const uiuxAgent: Agent = {
  name: 'uiux',
  manifest: uiuxManifest,
  findTrigger(board) {
    const hasUiuxSpec = board.some((m) => m.kind === 'uiux-spec');
    if (hasUiuxSpec) return null;
    // Trigger on the latest feature request if no UI/UX spec exists yet.
    const feature = [...board].reverse().find((m) => m.kind === 'feature-request');
    return feature ?? null;
  },
  async act(trigger, board) {
    const systemPrompt = `
You are the UI/UX Design Agent in a multi-agent engineering environment.

Your capability manifest:
- Name: ${uiuxManifest.name}
- Role: ${uiuxManifest.role}
- Description: ${uiuxManifest.description}
- Inputs: ${uiuxManifest.inputs.join(', ')}
- Outputs: ${uiuxManifest.outputs.join(', ')}
- Trigger signatures: ${uiuxManifest.triggerSignatures.join('; ')}
- Local policy: ${uiuxManifest.localPolicy}

You are handed a feature request from the global message board.
Produce a clear, implementation-ready UI/UX specification that frontend and backend engineer agents can act on.

Your output should be:
- A short summary of the feature from the user point-of-view
- Primary user flows, as bullet points or numbered steps
- Screen and component list, with responsibilities
- State and data that must be surfaced in the UI
- UX considerations (validation, loading, error, empty states)
- Open questions or assumptions (if any)

Write in concise, structured markdown. Do not include any code, only specifications.
`;

    const userPrompt = `
Feature request from the board (message #${trigger.id}):
${trigger.content}

Use the manifest and policies above to draft the UI/UX specification.
`;

    const result = streamText({
      model: anthropic('claude-3-haiku-20240307'),
      system: systemPrompt.trim(),
      messages: [{ role: 'user', content: userPrompt.trim() }],
    });

    let spec = '';
    process.stdout.write('\n[ui/ux agent] Generating UI/UX spec...\n');
    for await (const delta of result.textStream) {
      spec += delta;
      process.stdout.write(delta);
    }
    process.stdout.write('\n\n[ui/ux agent] UI/UX spec posted to the message board.\n\n');

    return createMessage('uiux', 'uiux-spec', spec.trim());
  },
};

const frontendAgent: Agent = {
  name: 'frontend',
  manifest: frontendManifest,
  findTrigger(board) {
    const hasFrontendPlan = board.some((m) => m.kind === 'frontend-plan');
    if (hasFrontendPlan) return null;
    const uiuxSpec = [...board].reverse().find((m) => m.kind === 'uiux-spec');
    return uiuxSpec ?? null;
  },
  async act(trigger, board) {
    const feature = board.find((m) => m.kind === 'feature-request');

    const systemPrompt = `
You are the Frontend Engineer Agent in a multi-agent engineering environment.

Your capability manifest:
- Name: ${frontendManifest.name}
- Role: ${frontendManifest.role}
- Description: ${frontendManifest.description}
- Inputs: ${frontendManifest.inputs.join(', ')}
- Outputs: ${frontendManifest.outputs.join(', ')}
- Trigger signatures: ${frontendManifest.triggerSignatures.join('; ')}
- Local policy: ${frontendManifest.localPolicy}

You receive UI/UX specs from the UI/UX agent and the original feature request.
Produce an actionable frontend implementation plan.

Your output should be:
- Technical summary of the UI to build
- Component breakdown and responsibilities
- Suggested routing/navigation structure
- State management approach
- Data fetching strategy and API surface needed from backend
- Edge cases and validation in the UI
`;

    const userPrompt = `
Original feature request:
${feature ? feature.content : '(not found on board)'}

UI/UX specification (message #${trigger.id}):
${trigger.content}

Use the manifest and policies above to create a frontend implementation plan.
`;

    const result = streamText({
      model: anthropic('claude-3-haiku-20240307'),
      system: systemPrompt.trim(),
      messages: [{ role: 'user', content: userPrompt.trim() }],
    });

    let plan = '';
    process.stdout.write('\n[frontend agent] Generating frontend implementation plan...\n');
    for await (const delta of result.textStream) {
      plan += delta;
      process.stdout.write(delta);
    }
    process.stdout.write(
      '\n\n[frontend agent] Frontend implementation plan posted to the message board.\n\n',
    );

    return createMessage('frontend', 'frontend-plan', plan.trim());
  },
};

const backendAgent: Agent = {
  name: 'backend',
  manifest: backendManifest,
  findTrigger(board) {
    const hasBackendPlan = board.some((m) => m.kind === 'backend-plan');
    if (hasBackendPlan) return null;
    const uiuxSpec = [...board].reverse().find((m) => m.kind === 'uiux-spec');
    return uiuxSpec ?? null;
  },
  async act(trigger, board) {
    const feature = board.find((m) => m.kind === 'feature-request');

    const systemPrompt = `
You are the Backend Engineer Agent in a multi-agent engineering environment.

Your capability manifest:
- Name: ${backendManifest.name}
- Role: ${backendManifest.role}
- Description: ${backendManifest.description}
- Inputs: ${backendManifest.inputs.join(', ')}
- Outputs: ${backendManifest.outputs.join(', ')}
- Trigger signatures: ${backendManifest.triggerSignatures.join('; ')}
- Local policy: ${backendManifest.localPolicy}

You receive UI/UX specs from the UI/UX agent and the original feature request.
Produce an actionable backend design and implementation plan.

Your output should be:
- Proposed APIs and endpoints (just descriptions, not code)
- Data models and relationships
- Auth and permission considerations
- Integration points with external systems (if any)
- Performance/scaling considerations
- Open questions or assumptions
`;

    const userPrompt = `
Original feature request:
${feature ? feature.content : '(not found on board)'}

UI/UX specification (message #${trigger.id}):
${trigger.content}

Use the manifest and policies above to create a backend design and implementation plan.
`;

    const result = streamText({
      model: anthropic('claude-3-haiku-20240307'),
      system: systemPrompt.trim(),
      messages: [{ role: 'user', content: userPrompt.trim() }],
    });

    let plan = '';
    process.stdout.write('\n[backend agent] Generating backend implementation plan...\n');
    for await (const delta of result.textStream) {
      plan += delta;
      process.stdout.write(delta);
    }
    process.stdout.write(
      '\n\n[backend agent] Backend implementation plan posted to the message board.\n\n',
    );

    return createMessage('backend', 'backend-plan', plan.trim());
  },
};

const agents: Agent[] = [uiuxAgent, frontendAgent, backendAgent];

async function runMultiAgentPipeline(initialRequest: string) {
  const board: BoardMessage[] = [];

  // Seed the board with the user's feature request.
  const featureMessage = createMessage('user', 'feature-request', initialRequest.trim());
  board.push(featureMessage);

  process.stdout.write(
    `\n[system] Feature request posted to the message board as message #${featureMessage.id}.\n`,
  );

  let progressed = true;

  // Simple asynchronous-style polling loop: each agent inspects the same global board
  // and decides whether to act based on its manifest-defined triggers.
  while (progressed) {
    progressed = false;

    for (const agent of agents) {
      const trigger = agent.findTrigger(board);
      if (!trigger) continue;

      progressed = true;
      const newMessage = await agent.act(trigger, board);
      board.push(newMessage);
    }
  }

  process.stdout.write('\n[system] Multi-agent pipeline finished for this feature request.\n\n');

  // Print a compact view of the final message board.
  process.stdout.write('[message board]\n');
  for (const msg of board) {
    const ts = msg.createdAt.toISOString();
    process.stdout.write(
      `#${msg.id} [${ts}] author=${msg.author} kind=${msg.kind}\n${msg.content}\n\n`,
    );
  }
}

async function main() {
  while (true) {
    const input = await terminal.question(
      'Enter a feature request for the multi-agent environment (or type "exit" to quit): ',
    );

    const trimmed = input.trim();
    if (!trimmed) {
      continue;
    }

    if (trimmed.toLowerCase() === 'exit') {
      break;
    }

    await runMultiAgentPipeline(trimmed);
  }

  terminal.close();
}

main().catch(console.error);