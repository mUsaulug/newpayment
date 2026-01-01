package ulug.musa.posclient.api;

import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;
import ulug.musa.posclient.service.StreamService;

@RestController
public class StreamController {
    private final StreamService streamService;

    public StreamController(StreamService streamService) {
        this.streamService = streamService;
    }

    @GetMapping(path = "/pos-client/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter stream() {
        return streamService.subscribe();
    }
}
