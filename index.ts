import { anthropic } from '@ai-sdk/anthropic';
import { streamText } from 'ai';
import 'dotenv/config';
import * as readline from 'node:readline/promises';

const terminal = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const POLL_INTERVAL_MS = 1_000;
const MAX_GLOBAL_CYCLES = 20;
const MAX_AGENT_ITERATIONS = 3;

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

interface BlackboardState {
  featureRequest?: BoardMessage;
  uiuxSpec?: BoardMessage;
  frontendPlan?: BoardMessage;
  backendPlan?: BoardMessage;
}

interface Agent {
  name: AgentName;
  manifest: CapabilityManifest;
  maxIterations: number;
  iterations: number;
  lastSummary?: string;
  /**
   * Inspect the global message board and decide whether to act.
   * Returns the message this agent wants to respond to, or null.
   */
  findTrigger(board: BoardMessage[], blackboard: BlackboardState): BoardMessage | null;
  /**
   * Run the agent's "act" function with the given trigger message
   * and produce a new message to be posted to the board.
   */
  act(
    trigger: BoardMessage,
    board: BoardMessage[],
    blackboard: BlackboardState,
  ): Promise<BoardMessage>;
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

function summarizeContent(content: string, maxLength = 200): string {
  const firstLine = content.split('\n').find((line) => line.trim().length > 0) ?? '';
  const trimmed = firstLine.trim();
  if (trimmed.length <= maxLength) return trimmed;
  return `${trimmed.slice(0, maxLength - 3)}...`;
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
  maxIterations: MAX_AGENT_ITERATIONS,
  iterations: 0,
  findTrigger(board, blackboard) {
    const feature = blackboard.featureRequest ?? [...board].reverse().find((m) => m.kind === 'feature-request');
    if (!feature) return null;

    const mySpecs = board.filter((m) => m.author === 'uiux' && m.kind === 'uiux-spec');

    // First iteration: respond directly to the feature request.
    if (mySpecs.length === 0) return feature;

    // Subsequent iterations: optionally refine the last spec,
    // up to maxIterations.
    if (this.iterations >= this.maxIterations) return null;
    return mySpecs[mySpecs.length - 1];
  },
  async act(trigger, board, _blackboard) {
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
      model: anthropic('claude-3-haiku-20240307') as any,
      system: systemPrompt.trim(),
      messages: [{ role: 'user', content: userPrompt.trim() }],
    });

    let spec = '';
    for await (const delta of result.textStream) {
      spec += delta;
    }

    const summary = summarizeContent(spec);
    uiuxAgent.lastSummary = summary;
    process.stdout.write(`\n[ui/ux agent] UI/UX spec updated: ${summary}\n\n`);

    return createMessage('uiux', 'uiux-spec', spec.trim());
  },
};

const frontendAgent: Agent = {
  name: 'frontend',
  manifest: frontendManifest,
  maxIterations: MAX_AGENT_ITERATIONS,
  iterations: 0,
  findTrigger(board, blackboard) {
    const uiuxSpec =
      blackboard.uiuxSpec ?? [...board].reverse().find((m) => m.kind === 'uiux-spec');
    if (!uiuxSpec) return null;

    const myPlans = board.filter((m) => m.author === 'frontend' && m.kind === 'frontend-plan');

    // First iteration: respond to the UI/UX spec.
    if (myPlans.length === 0) return uiuxSpec;

    // Subsequent iterations: refine the last frontend plan,
    // for example based on any new messages on the board.
    if (this.iterations >= this.maxIterations) return null;
    return myPlans[myPlans.length - 1];
  },
  async act(trigger, board, blackboard) {
    const feature = blackboard.featureRequest ?? board.find((m) => m.kind === 'feature-request');

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
      model: anthropic('claude-3-haiku-20240307') as any,
      system: systemPrompt.trim(),
      messages: [{ role: 'user', content: userPrompt.trim() }],
    });

    let plan = '';
    for await (const delta of result.textStream) {
      plan += delta;
    }

    const summary = summarizeContent(plan);
    frontendAgent.lastSummary = summary;
    process.stdout.write(`\n[frontend agent] Plan updated: ${summary}\n\n`);

    return createMessage('frontend', 'frontend-plan', plan.trim());
  },
};

const backendAgent: Agent = {
  name: 'backend',
  manifest: backendManifest,
  maxIterations: MAX_AGENT_ITERATIONS,
  iterations: 0,
  findTrigger(board, blackboard) {
    const uiuxSpec =
      blackboard.uiuxSpec ?? [...board].reverse().find((m) => m.kind === 'uiux-spec');
    if (!uiuxSpec) return null;

    const myPlans = board.filter((m) => m.author === 'backend' && m.kind === 'backend-plan');

    // First iteration: respond to the UI/UX spec.
    if (myPlans.length === 0) return uiuxSpec;

    // Subsequent iterations: refine the last backend plan.
    if (this.iterations >= this.maxIterations) return null;
    return myPlans[myPlans.length - 1];
  },
  async act(trigger, board, blackboard) {
    const feature = blackboard.featureRequest ?? board.find((m) => m.kind === 'feature-request');

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
      model: anthropic('claude-3-haiku-20240307') as any,
      system: systemPrompt.trim(),
      messages: [{ role: 'user', content: userPrompt.trim() }],
    });

    let plan = '';
    for await (const delta of result.textStream) {
      plan += delta;
    }

    const summary = summarizeContent(plan);
    backendAgent.lastSummary = summary;
    process.stdout.write(`\n[backend agent] Plan updated: ${summary}\n\n`);

    return createMessage('backend', 'backend-plan', plan.trim());
  },
};

const agents: Agent[] = [uiuxAgent, frontendAgent, backendAgent];

async function runMultiAgentPipeline(initialRequest: string) {
  const board: BoardMessage[] = [];
  const blackboard: BlackboardState = {};

  // Seed the board with the user's feature request.
  const featureMessage = createMessage('user', 'feature-request', initialRequest.trim());
  board.push(featureMessage);
  blackboard.featureRequest = featureMessage;

  process.stdout.write(
    `\n[system] Feature request posted to the message board as message #${featureMessage.id}.\n`,
  );

  let idleCycles = 0;

  // Asynchronous-style polling loop: each agent repeatedly inspects the same global
  // board, decides whether to act, and can refine its own outputs over multiple
  // iterations. The loop terminates once no agent acts for several cycles or we
  // hit safety limits.
  for (let cycle = 1; cycle <= MAX_GLOBAL_CYCLES; cycle++) {
    process.stdout.write(`\n[system] Poll cycle ${cycle}...\n`);

    let progressedThisCycle = false;

    for (const agent of agents) {
      const trigger = agent.findTrigger(board, blackboard);
      if (!trigger) continue;
      if (agent.iterations >= agent.maxIterations) continue;

      agent.iterations += 1;
      progressedThisCycle = true;

      const newMessage = await agent.act(trigger, board, blackboard);
      board.push(newMessage);

      // Update the shared blackboard with the latest outputs for each agent.
      if (newMessage.kind === 'uiux-spec') {
        blackboard.uiuxSpec = newMessage;
      } else if (newMessage.kind === 'frontend-plan') {
        blackboard.frontendPlan = newMessage;
      } else if (newMessage.kind === 'backend-plan') {
        blackboard.backendPlan = newMessage;
      }
    }

    if (!progressedThisCycle) {
      idleCycles += 1;
      if (idleCycles >= 3) {
        process.stdout.write(
          '\n[system] No agent activity for several cycles; stopping the multi-agent loop.\n\n',
        );
        break;
      }
    } else {
      idleCycles = 0;
    }

    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }

  process.stdout.write('\n[system] Multi-agent pipeline reached its stopping condition.\n\n');

  // Print brief, user-friendly summaries of what each agent did for this request.
  process.stdout.write('[agent summaries]\n');

  const agentNames: AgentName[] = ['uiux', 'frontend', 'backend'];

  for (const name of agentNames) {
    const agent = agents.find((a) => a.name === name)!;
    const agentMessages = board.filter((m) => m.author === name);
    const iterations = agentMessages.length;

    if (iterations === 0 || !agent.lastSummary) {
      process.stdout.write(`- ${name}: no output\n`);
      continue;
    }

    process.stdout.write(
      `- ${name} (iterations: ${iterations}, last kind: ${
        agentMessages[agentMessages.length - 1].kind
      }): ${agent.lastSummary}\n`,
    );
  }

  process.stdout.write('\n');
}

async function main() {
  while (true) {
    // const input = await terminal.question(
    //   'Enter a feature request for the multi-agent environment (or type "exit" to quit): ',
    // );
    const input = `Build a weekly dashboard that tracks active users for our system.`;

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